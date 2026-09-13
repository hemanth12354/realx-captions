 import streamlit as st
import whisper
import subprocess
import os
import re
import json
import tempfile

st.set_page_config(page_title="Real-X Auto Captions", page_icon="🎬", layout="wide")

ROLES = ["setup", "emphasis", "trailing"]
ZONES = ["upper", "middle", "lower"]
ZONE_Y = {"upper": 0.36, "middle": 0.5, "lower": 0.7}

DEFAULT_ROLE_STYLE = {
    "setup":    {"color": "#FFFFFF", "outline": "#000000", "size": "medium"},
    "emphasis": {"color": "#CCFF66", "outline": "#000000", "size": "large"},
    "trailing": {"color": "#FFFFFF", "outline": "#000000", "size": "medium"},
}
SIZE_PX = {"small": 26, "medium": 38, "large": 60}


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

    style = json.loads(json.dumps(DEFAULT_ROLE_STYLE))  # deep copy
    for role in ROLES:
        if role in parsed and isinstance(parsed[role], dict):
            style[role].update({k: v for k, v in parsed[role].items() if k in ("color", "outline", "size")})
    return style


def design_caption_layout(api_key: str, words: list) -> list:
    client = gemini_client(api_key)
    transcript = json.dumps(words)

    prompt = (
        "You are designing captions for a short-form vertical video (like a viral "
        "Instagram Reel), in the style where captions are split into short beats. "
        "Each beat can have up to 3 simultaneous text elements: 'setup' (context "
        "words, smaller), 'emphasis' (the 1-3 most important/punchy words in that "
        "beat, bigger and bolder), and 'trailing' (any words after the emphasis in "
        "that beat, smaller again). Not every beat needs all three roles.\n\n"
        "Here is the word-by-word transcript with timestamps (start/end in seconds):\n"
        f"{transcript}\n\n"
        "Group these words into beats and assign roles. Reply with ONLY a JSON "
        "array, no other text, where each item is: "
        '{"text": "...", "start": <number>, "end": <number>, "role": "setup|emphasis|trailing", '
        '"zone": "upper|middle|lower"}. '
        "Use the exact start/end times from the given words (combine consecutive "
        "words' times when grouping them into one element). Vary the zone across "
        "consecutive beats for visual variety. Keep emphasis elements short (1-3 words)."
    )

    response = client.models.generate_content(model="gemini-3.6-flash", contents=[prompt])
    match = re.search(r"\[.*\]", response.text.strip(), re.DOTALL)
    if not match:
        raise ValueError("Could not design a caption layout.")
    elements = json.loads(match.group(0))

    cleaned = []
    for el in elements:
        role = el.get("role") if el.get("role") in ROLES else "setup"
        zone = el.get("zone") if el.get("zone") in ZONES else "middle"
        cleaned.append({
            "text": str(el.get("text", "")).strip(),
            "start": float(el.get("start", 0)),
            "end": float(el.get("end", 0)),
            "role": role,
            "zone": zone,
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
            f"Style: {role},DejaVu Sans,{size},{primary},{primary},{outline},"
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
        y = int(video_h * ZONE_Y.get(el["zone"], 0.5))
        start = format_ass_timestamp(el["start"])
        end = format_ass_timestamp(el["end"])
        text = el["text"].replace("\n", " ")
        tag = f"{{\\an5\\pos({x},{y})\\fscx70\\fscy70\\t(0,150,\\fscx100\\fscy100)\\fad(60,0)}}"
        lines.append(f"Dialogue: 0,{start},{end},{el['role']},,0,0,0,,{tag}{text}\n")
    return "".join(lines)


def burn_layered(video_path: str, elements: list, output_path: str, role_style: dict, ass_path: str):
    video_w, video_h = get_video_resolution(video_path)
    ass_content = build_layered_ass(elements, video_w, video_h, role_style)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={ass_path}",
        "-c:a", "copy", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------- UI ----------------

st.title("🎬 Real-X Auto Captions")
st.write(
    "Upload a video. It transcribes the speech, then AI designs a multi-layer "
    "animated caption layout (setup / emphasis / trailing words), optionally "
    "matched to a reference video's style."
)

if "role_style" not in st.session_state:
    st.session_state.role_style = json.loads(json.dumps(DEFAULT_ROLE_STYLE))
if "words" not in st.session_state:
    st.session_state.words = None
if "elements" not in st.session_state:
    st.session_state.elements = None
if "main_video_bytes" not in st.session_state:
    st.session_state.main_video_bytes = None

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.warning("No Gemini API key found in Settings → Secrets. Reference-style matching and AI layout design need it.")

col1, col2 = st.columns(2)
with col1:
    main_video = st.file_uploader("Your video (to caption)", type=["mp4", "mov", "m4v"], key="main_video_uploader")
with col2:
    reference_video = st.file_uploader(
        "Reference video (optional — copy its caption style)",
        type=["mp4", "mov", "m4v"], key="ref_video_uploader",
    )

if reference_video is not None and api_key and st.button("Analyze reference style"):
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.mp4")
        with open(ref_path, "wb") as f:
            f.write(reference_video.read())
        frame_path = os.path.join(tmp, "frame.jpg")
        try:
            with st.spinner("Reading the reference video's caption style..."):
                extract_frame(ref_path, frame_path)
                st.session_state.role_style = analyze_reference_style(api_key, frame_path)
            st.success(f"Style detected: {st.session_state.role_style}")
        except Exception as e:
            st.error(f"Couldn't analyze the reference video: {e}")

st.subheader("Caption style per role")
rs = st.session_state.role_style
role_cols = st.columns(3)
for i, role in enumerate(ROLES):
    with role_cols[i]:
        st.markdown(f"**{role.capitalize()}**")
        rs[role]["color"] = st.color_picker(f"{role} text color", rs[role]["color"], key=f"{role}_color")
        rs[role]["outline"] = st.color_picker(f"{role} outline", rs[role]["outline"], key=f"{role}_outline")
        rs[role]["size"] = st.selectbox(f"{role} size", ["small", "medium", "large"],
                                         index=["small", "medium", "large"].index(rs[role]["size"]),
                                         key=f"{role}_size")

st.divider()

if main_video is not None and st.button("Transcribe"):
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(main_video.read())
        st.session_state.main_video_bytes = open(video_path, "rb").read()

        with st.spinner("Loading model and transcribing (first run takes longer)..."):
            model = load_whisper_model()
            words, language = transcribe_words(model, video_path)
        st.session_state.words = words
        st.session_state.elements = None
        st.info(f"Detected language: {language}.")

if st.session_state.words and api_key:
    if st.button("Design caption layout with AI"):
        with st.spinner("Designing the multi-layer caption layout..."):
            try:
                st.session_state.elements = design_caption_layout(api_key, st.session_state.words)
            except Exception as e:
                st.error(f"Couldn't design a layout: {e}")

if st.session_state.elements:
    st.subheader("Edit caption layout")
    st.caption("Adjust text, role, or zone for any beat before rendering.")
    edited = st.data_editor(
        st.session_state.elements,
        column_config={
            "text": st.column_config.TextColumn("Text"),
            "start": st.column_config.NumberColumn("Start (s)", disabled=True),
            "end": st.column_config.NumberColumn("End (s)", disabled=True),
            "role": st.column_config.SelectboxColumn("Role", options=ROLES),
            "zone": st.column_config.SelectboxColumn("Zone", options=ZONES),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="layout_editor",
    )

    if st.button("Generate captioned video"):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(st.session_state.main_video_bytes)

            output_path = os.path.join(tmp, "output.mp4")
            ass_path = os.path.join(tmp, "captions.ass")
            with st.spinner("Rendering animated captions onto video..."):
                burn_layered(video_path, edited, output_path, st.session_state.role_style, ass_path)

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.success("Done!")
            st.video(video_bytes)
            st.download_button(
                "Download captioned video",
                data=video_bytes,
                file_name="captioned_video.mp4",
                mime="video/mp4",
            )
elif st.session_state.words and not api_key:
    st.error("AI layout design needs the Gemini API key in Settings → Secrets.")
