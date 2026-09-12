import streamlit as st
import whisper
import subprocess
import os
import re
import json
import tempfile

st.set_page_config(page_title="Real-X Auto Captions", page_icon="🎬", layout="wide")

DEFAULT_STYLE = {
    "position": "bottom",
    "text_color": "#FFFFFF",
    "outline_color": "#000000",
    "highlight_color": "#FFD700",
    "font_size": "medium",
}

ALIGNMENT_MAP = {"bottom": 2, "middle": 5, "top": 8}
FONT_SIZE_MAP = {"small": 18, "medium": 24, "large": 32}


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def hex_to_ass_color(hex_str: str) -> str:
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        hex_str = "FFFFFF"
    rr, gg, bb = hex_str[0:2], hex_str[2:4], hex_str[4:6]
    return f"&H00{bb}{gg}{rr}&".upper()


def format_srt_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


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


def words_to_srt(words) -> str:
    lines = []
    for idx, w in enumerate(words, start=1):
        start = format_srt_timestamp(w["start"])
        end = format_srt_timestamp(w["end"])
        lines.append(f"{idx}\n{start} --> {end}\n{w['word']}\n")
    return "\n".join(lines)


def extract_frame(video_path: str, out_path: str, at_seconds: float = 1.0):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(at_seconds), "-i", video_path, "-vframes", "1", out_path],
        check=True, capture_output=True,
    )


def analyze_reference_style(api_key: str, frame_path: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "Look at the video caption/subtitle style visible in this image. "
        "Reply with ONLY a JSON object, no other text, with these exact keys: "
        '"position" (one of: top, middle, bottom), '
        '"text_color" (hex code like #FFFFFF for the main caption text color), '
        '"outline_color" (hex code for the text outline/border color), '
        '"highlight_color" (hex code for any emphasized/highlighted word color, '
        'or the same as text_color if no highlight is visible), '
        '"font_size" (one of: small, medium, large, based on how large the text looks relative to the frame).'
    )

    response = client.models.generate_content(
        model="model="gemini-3.6-flash",
        contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
    )

    text = response.text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Could not parse a style from the reference video.")
    parsed = json.loads(match.group(0))

    style = DEFAULT_STYLE.copy()
    style.update({k: v for k, v in parsed.items() if k in DEFAULT_STYLE})
    return style


def burn_captions(video_path: str, srt_path: str, output_path: str, style: dict):
    alignment = ALIGNMENT_MAP.get(style["position"], 2)
    font_size = FONT_SIZE_MAP.get(style["font_size"], 24)
    primary = hex_to_ass_color(style["text_color"])
    outline = hex_to_ass_color(style["outline_color"])

    force_style = (
        f"FontName=Arial,FontSize={font_size},PrimaryColour={primary},"
        f"OutlineColour={outline},BorderStyle=1,Outline=2,Alignment={alignment}"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='{force_style}'",
        "-c:a", "copy", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------- UI ----------------

st.title("🎬 Real-X Auto Captions")
st.write("Upload a video to caption. Optionally upload a reference video to match its caption style.")

col1, col2 = st.columns(2)
with col1:
    main_video = st.file_uploader("Your video (to caption)", type=["mp4", "mov", "m4v"], key="main_video")
with col2:
    reference_video = st.file_uploader(
        "Reference video (optional — copy its caption style)",
        type=["mp4", "mov", "m4v"], key="ref_video",
    )

if "style" not in st.session_state:
    st.session_state.style = DEFAULT_STYLE.copy()
if "words" not in st.session_state:
    st.session_state.words = None

if reference_video is not None and st.button("Analyze reference style"):
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("No Gemini API key found in secrets. Add GEMINI_API_KEY in your app's Settings → Secrets.")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.mp4")
            with open(ref_path, "wb") as f:
                f.write(reference_video.read())
            frame_path = os.path.join(tmp, "frame.jpg")
            try:
                with st.spinner("Grabbing a frame and asking Gemini about the style..."):
                    extract_frame(ref_path, frame_path)
                    st.session_state.style = analyze_reference_style(api_key, frame_path)
                st.success(f"Style detected: {st.session_state.style}")
            except Exception as e:
                st.error(f"Couldn't analyze the reference video: {e}")

st.subheader("Caption style")
s = st.session_state.style
c1, c2, c3 = st.columns(3)
with c1:
    s["position"] = st.selectbox("Position", ["bottom", "middle", "top"],
                                  index=["bottom", "middle", "top"].index(s["position"]))
    s["font_size"] = st.selectbox("Font size", ["small", "medium", "large"],
                                   index=["small", "medium", "large"].index(s["font_size"]))
with c2:
    s["text_color"] = st.color_picker("Text color", s["text_color"])
    s["outline_color"] = st.color_picker("Outline color", s["outline_color"])
with c3:
    s["highlight_color"] = st.color_picker("Highlight color (future use)", s["highlight_color"])

st.divider()

if main_video is not None and st.button("Transcribe"):
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(main_video.read())
        st.session_state["main_video_bytes"] = open(video_path, "rb").read()

        with st.spinner("Loading model and transcribing (first run takes longer)..."):
            model = load_whisper_model()
            words, language = transcribe_words(model, video_path)
        st.session_state.words = words
        st.info(f"Detected language: {language}. Review/edit the words below, then generate the video.")

if st.session_state.words:
    st.subheader("Edit captions")
    st.caption("Fix any words the transcription got wrong. Timing stays the same.")
    edited = st.data_editor(
        st.session_state.words,
        column_config={
            "word": st.column_config.TextColumn("Word"),
            "start": st.column_config.NumberColumn("Start (s)", disabled=True),
            "end": st.column_config.NumberColumn("End (s)", disabled=True),
        },
        num_rows="fixed",
        use_container_width=True,
        key="editor",
    )

    if st.button("Generate captioned video"):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(st.session_state["main_video_bytes"])

            srt_path = os.path.join(tmp, "captions.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(words_to_srt(edited))

            output_path = os.path.join(tmp, "output.mp4")
            with st.spinner("Burning captions onto video..."):
                burn_captions(video_path, srt_path, output_path, st.session_state.style)

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
             
 
