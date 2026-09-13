import streamlit as st
import whisper
import subprocess
import os
import re
import json
import tempfile

st.set_page_config(page_title="Real-X Auto Captions", page_icon="🎬", layout="wide")

ROLES = ["setup", "emphasis", "trailing"]

DEFAULT_ROLE_STYLE = {
    "setup":    {"color": "#FFFFFF", "outline": "#000000", "size": "medium"},
    "emphasis": {"color": "#CCFF66", "outline": "#000000", "size": "large"},
    "trailing": {"color": "#FFFFFF", "outline": "#000000", "size": "medium"},
}
SIZE_PX = {"small": 26, "medium": 38, "large": 60}
FONT_NAME = "Poppins"


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def hex_to_ass_color(hex_str: str) -> str:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        hex_str = "FFFFFF"
    rr, gg, bb = hex_str[0:2], hex_str[2:4], hex_str[4:6]
    return f"&H00{bb}{gg}{rr}&".upper()


def format_ass_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{hrs}:{mins:02}:{secs:02}.{cs:02}"


def transcribe_words(model, video_path: str):
    result = model.transcribe(video_path, word_timestamps=True)
    words = []
    for segment in result["segments"]:
        for word in segment.get("words", []):
            text = word["word"].strip()
            if not text:
                continue
            words.append({"word": text, "start": round(word["start"], 2), "end": round(word["end"], 2)})
    return words, result.get("language", "unknown")


def get_video_resolution(video_path: str):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        check=True, capture_output=True, text=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def extract_frame(video_path: str, out_path: str, at_seconds: float = 1.0):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(at_seconds), "-i", video_path, "-vframes", "1", out_path],
        check=True, capture_output=True,
    )


def gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def analyze_reference_style(api_key: str, frame_path: str) -> dict:
    from google.genai import types

    client = gemini_client(api_key)
    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "Look at the video caption/subtitle style in this image. Viral short-form "
        "reels often show up to 3 kinds of caption text at once: a 'setup' phrase "
        "(context, usually smaller/plainer), an 'emphasis' word or two (usually "
        "bigger, bolder, brightly colored), and 'trailing' words (smaller, after "
        "the emphasis). Reply with ONLY a JSON object, no other text, shaped like: "
        '{"setup": {"color": "#hex", "outline": "#hex", "size": "small|medium|large"}, '
        '"emphasis": {"color": "#hex", "outline": "#hex", "size": "small|medium|large"}, '
        '"trailing": {"color": "#hex", "outline": "#hex", "size": "small|medium|large"}} '
        "Base this on what's actually visible. If only one caption style is visible, "
        "use it for 'setup' and 'trailing', and pick a plausible brighter/bigger "
        "variant for 'emphasis'."
    )

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
    )
    match = re.search(r"\{.*\}", response.text.strip(), re.DOTALL)
    if not match:
        raise ValueError("Could not parse a style from the reference video.")
    parsed = json.loads(match.group(0))

    style = json.loads(json.dumps(DEFAULT_ROLE_STYLE))
    for role in ROLES:
        if role in parsed and isinstance(parsed[role], dict):
            style[role].update({k: v for k, v in parsed[role].items() if k in ("color", "outline", "size")})
    return style


def design_caption_layout(api_key: str, words: list) -> list:
    client = gemini_client(api_key)
    transcript = json.dumps(words)

    prompt = (
        "You are designing captions for a short-form vertical video (like a viral "
        "Instagram Reel), split into short beats. Each beat can have up to 3 "
        "simultaneous text elements: 'setup' (context words, smaller), 'emphasis' "
        "(the 1-3 most important/punchy words, bigger and bolder), and 'trailing' "
        "(words after the emphasis, smaller again). Not every beat needs all three.\n\n"
        "Here is the word-by-word transcript with timestamps (start/end in seconds):\n"
        f"{transcript}\n\n"
        "Group these words into beats and assign roles. Reply with ONLY a JSON "
        "array, no other text, where each item is: "
        '{"text": "...", "start": <number>, "end": <number>, "role": "setup|emphasis|trailing", '
        '"y_percent": <number 0-100, vertical position from top of frame>}. '
        "Use the exact start/end times from the given words (combine consecutive "
        "words' times when grouping them into one element). Vary y_percent across "
        "consecutive beats for visual variety (e.g. 35, 50, 65). Keep emphasis "
        "elements short (1-3 words)."
    )

    response = client.models.generate_content(model="gemini-3.6-flash", contents=[prompt])
    match = re.search(r"\[.*\]", response.text.strip(), re.DOTALL)
    if not match:
        raise ValueError("Could not design a caption layout.")
    elements = json.loads(match.group(0))

    cleaned = []
    for el in elements:
        role = el.get("role") if el.get("role") in ROLES else "setup"
        try:
            y_percent = float(el.get("y_percent", 50))
        except (TypeError, ValueError):
            y_percent = 50.0
        y_percent = max(0.0, min(100.0, y_percent))
        cleaned.append({
            "text": str(el.get("text", "")).strip(),
            "start": float(el.get("start", 0)),
            "end": float(el.get("end", 0)),
            "role": role,
            "y_percent": y_percent,
        })
    return [e for e in cleaned if e["text"]]


def build_layered_ass(elements: list, video_w: int, video_h: int, role_style: dict) -> str:
    styles_block = []
    for role in ROLES:
        st_cfg = role_style[role]
        primary = hex_to_ass_color(st_cfg["color"])
        outline = hex_to_ass_color(st_cfg["outline"])
        size = SIZE_PX.get(st_cfg["size"], 38)
        styles_block.append(
            f"Style: {role},{FONT_NAME},{size},{primary},{primary},{outline},"
            f"&H00000000,-1,0,0,0,100,100,0,0,1,3,0,5,10,10,10,1"
        )

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles_block)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    x = video_w // 2
    for el in elements:
        y = int(video_h * (el["y_percent"] / 100.0))
        start = format_ass_timestamp(el["start"])
        end = format_ass_timestamp(el["end"])
        text = el["text"].replace("\n", " ")
        tag = f"{{\\an5\\pos({x},{y})\\fscx70\\fscy70\\t(0,150,\\fscx100\\fscy100)\\fad(60,0)}}"
        lines.append(f"Dialogue: 0,{start},{end},{el['role']},,0,0,0,,{tag}{text}\n")
    return "".join(lines)


def burn_layered(video_path: str, elements: list, output_path: str, role_style: dict, ass_path: str, fonts_dir: str):
    video_w, video_h = get_video_resolution(video_path)
    ass_content = build_layered_ass(elements, video_w, video_h, role_style)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={ass_path}:fontsdir={fonts_dir}",
        "-c:a", "copy", output_path,
    ]
