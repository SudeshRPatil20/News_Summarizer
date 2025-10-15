import streamlit as st
from pathlib import Path
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave, json, os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import validators
import re
import google.generativeai as genai

SetLogLevel(-1)

load_dotenv()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)

DEFAULT_MODEL_PATH = r"vosk-model-small-en-us-0.15"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Audio / YouTube / Website Summarizer", layout="wide")
st.title("📺 Audio / YouTube / Website Summarizer (Vosk + Google AI / Local)")

st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Vosk Model Path", DEFAULT_MODEL_PATH)
max_summary_sentences = st.sidebar.number_input("Max sentences in summary", min_value=1, max_value=10, value=3)

st.subheader("Input Options")
audio_file = st.file_uploader("🎧 Upload audio file (WAV/MP3)", type=["wav", "mp3"])
youtube_url = st.text_input("📹 Or enter YouTube URL")
website_url = st.text_input("🌐 Or enter Website URL (e.g. news article)")

def summarize_with_google(transcript: str, max_sentences: int = 3) -> str:
    prompt = f"Summarize the following text in at most {max_sentences} sentences:\n\n{transcript}"
    response = genai.chat.create(
        model="gemini-flash-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.last.response

def summarize_locally(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:max_sentences])

@st.cache_resource(ttl=3600)
def load_vosk_model(path):
    model_folder = Path(path)
    if not model_folder.exists():
        st.error(f"Vosk model folder not found: {path}")
        st.stop()
    return Model(str(model_folder))

model = load_vosk_model(model_path)

def transcribe_audio(file_path, model):
    wf = wave.open(str(file_path), "rb")
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

def normalize_youtube_url(url):
    parsed = urlparse(url)
    if "youtube" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return f"https://www.youtube.com/watch?v={query['v'][0]}"
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

if st.button("🔍 Generate Summary"):
    transcript = ""

    if audio_file:
        st.info("Processing uploaded audio file...")
        audio_path = OUTPUT_DIR / audio_file.name
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())
        with st.spinner("Transcribing audio with Vosk..."):
            try:
                transcript = transcribe_audio(audio_path, model)
            except Exception as e:
                st.error(f"Audio transcription failed: {e}")

    elif youtube_url.strip():
        try:
            from langchain_community.document_loaders import YoutubeLoader
            st.info("Fetching YouTube transcript...")
            yt_url = normalize_youtube_url(youtube_url)
            loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True, language="en")
            with st.spinner("Loading YouTube transcript..."):
                docs = loader.load()
                transcript = " ".join([doc.page_content for doc in docs])
        except Exception as e:
            st.error(f"Failed to fetch YouTube transcript: {e}")

    elif website_url.strip():
        try:
            from langchain_community.document_loaders import UnstructuredURLLoader
            st.info("Extracting website text...")
            if validators.url(website_url):
                loader = UnstructuredURLLoader(urls=[website_url])
                with st.spinner("Loading website content..."):
                    docs = loader.load()
                    transcript = " ".join([doc.page_content for doc in docs])[:10000]
            else:
                st.error("Invalid URL format")
        except Exception as e:
            st.error(f"Failed to fetch website text: {e}")

    else:
        st.warning("Provide an input (Audio / YouTube / Website)")

    st.subheader("Transcript (preview)")
    if transcript:
        st.write(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
    else:
        st.warning("Transcript is empty. Cannot generate summary.")

    st.info("Generating summary...")
    try:
        if GOOGLE_API_KEY:
            summary = summarize_with_google(transcript, max_summary_sentences)
        else:
            summary = summarize_locally(transcript, max_summary_sentences)
        st.subheader("Summary")
        st.write(summary)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        summary = summarize_locally(transcript, max_summary_sentences)
        st.subheader("Summary (Local fallback)")
        st.write(summary)

