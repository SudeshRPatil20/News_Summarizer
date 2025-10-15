import streamlit as st
import subprocess
import tempfile
import os
import wave
import json
import time
from datetime import datetime
from vosk import Model, KaldiRecognizer, SetLogLevel
import google.genai as genai
from pathlib import Path
from pydub import AudioSegment
import google.generativeai as genaii

SetLogLevel(-1)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="📺 Monitor Agent Prototype", layout="wide")
st.title("📺 Fox News Monitor Agent")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genaii.configure(api_key=GOOGLE_API_KEY)

st.sidebar.header("Settings")
video_url = st.sidebar.text_input("Enter Fox News video stream/file path", "")
uploaded_file = st.sidebar.file_uploader("Or upload Fox News audio/video file", type=["wav","mp3","mp4"])
chunk_duration = st.sidebar.number_input("Transcript chunk (seconds)", 30, 300, 60)
max_summary_words = st.sidebar.number_input("Max words in summary", 5, 20, 15)
vosk_model_path = st.sidebar.text_input("Vosk Model Path", "vosk-model-small-en-us-0.15")

@st.cache_resource
def load_vosk_model(path):
    if not Path(path).exists():
        st.error(f"Vosk model not found at {path}")
        st.stop()
    return Model(path)

model = load_vosk_model(vosk_model_path)

def convert_to_wav(uploaded_file):
    temp_in = tempfile.NamedTemporaryFile(delete=False)
    temp_in.write(uploaded_file.read())
    temp_in.flush()
    sound = AudioSegment.from_file(temp_in.name)
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.export(temp_out.name, format="wav")
    return temp_out.name

def transcribe_audio(file_path, model):
    try:
        wf = wave.open(file_path, "rb")
    except EOFError:
        st.error("Invalid or unsupported audio file.")
        return ""
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))
    texts = [r.get("text", "").strip() for r in results if r.get("text")]
    return " ".join(texts).strip()

def summarize_with_gemini(transcript: str, max_words: int = 15) -> str:
    prompt = f"Summarize in max {max_words} words: {transcript}"
    try:
        response = genai.responses.create(
            model="gemini-flash-latest",
            input=prompt
        )
        return response.output_text.strip()
    except Exception as e:
        return f"(Summary failed: {e})"

def capture_audio_chunk(input_src, duration, out_path, uploaded=False):
    if uploaded:
        sound = AudioSegment.from_file(input_src)
        chunk = sound[:duration*1000]
        chunk.export(out_path, format="wav")
    else:
        command = [
            "ffmpeg", "-y",
            "-i", input_src,
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            out_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if st.button("🎬 Fetch Next Chunk"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_file = tmp.name

    if uploaded_file:
        wav_path = convert_to_wav(uploaded_file)
        capture_audio_chunk(wav_path, chunk_duration, wav_file, uploaded=True)
    elif video_url.strip():
        capture_audio_chunk(video_url.strip(), chunk_duration, wav_file)
    else:
        st.warning("Please upload a Fox News file or provide a video URL.")
        st.stop()

    transcript = transcribe_audio(wav_file, model)
    if transcript:
        summary = summarize_with_gemini(transcript, max_summary_words) if GOOGLE_API_KEY else transcript[:max_summary_words]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {"timestamp": timestamp, "transcript": transcript, "summary": summary}
        json_path = OUTPUT_DIR / f"{int(time.time())}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        st.subheader(f"🕒 {timestamp}")
        st.write("**Transcript (preview):**", transcript[:500] + "...")
        st.write("**Summary:**", summary)
    else:
        st.warning("No transcript extracted.")

    os.remove(wav_file)
