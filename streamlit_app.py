import streamlit as st
import whisper
import subprocess
import os
import tempfile

st.set_page_config(page_title="Real-X Auto Captions", page_icon="🎬")


@st.cache_resource
def load_model():
    return whisper.load_model("base")


def format_srt_timestamp(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"


def generate_srt(model, video_path: str):
    result = model.transcribe(video_path, word_timestamps=True)
    lines = []
    idx = 1
    for segment in result["segments"]:
        for word in segment.get("words", []):
            text = word["word"].strip()
            if not text:
                continue
            start = format_srt_timestamp(word["start"])
            end = format_srt_timestamp(word["end"])
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            idx += 1
    return "\n".join(lines), result.get("language", "unknown")


def burn_captions(video_path: str, srt_path: str, output_path: str):
    style = (
        "FontName=Arial,FontSize=22,PrimaryColour=&HFFFFFF&,"
        "OutlineColour=&H000000&,BorderStyle=1,Outline=2,Alignment=2"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"subtitles={srt_path}:force_style='{style}'",
        "-c:a", "copy", output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


st.title("🎬 Real-X Auto Captions")
st.write("Upload a video. It auto-detects the spoken language and burns word-timed captions onto it.")

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "m4v"])

if uploaded_file is not None:
    if st.button("Generate captions"):
        with st.spinner("Loading model (first run takes a minute)..."):
            model = load_model()

        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "input.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Transcribing..."):
                srt_content, language = generate_srt(model, video_path)

            if not srt_content.strip():
                st.error("No speech detected in this video.")
            else:
                st.info(f"Detected language: {language}")
                srt_path = os.path.join(tmp, "captions.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)

                output_path = os.path.join(tmp, "output.mp4")
                with st.spinner("Burning captions onto video..."):
                    burn_captions(video_path, srt_path, output_path)

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
