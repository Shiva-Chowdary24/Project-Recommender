import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import streamlit as st
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from faster_whisper import WhisperModel
from audio_recorder_streamlit import audio_recorder
import tempfile

# -----------------------------
# CONFIG
# -----------------------------
TOP_K = 4  # 1 best + 3 recommendations

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD FAISS + IMAGE PATHS
# -----------------------------
@st.cache_resource
def load_faiss():
    index = faiss.read_index("faiss.index")
    with open("image_paths.txt") as f:
        paths = [line.strip() for line in f]
    return index, paths

index, image_paths = load_faiss()

# -----------------------------
# LOAD CLIP
# -----------------------------
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

clip_model, clip_processor = load_clip()

# -----------------------------
# LOAD WHISPER
# -----------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")

whisper_model = load_whisper()

# -----------------------------
# TEXT ENCODING
# -----------------------------
def encode_text(text):
    with torch.no_grad():
        inputs = clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(device)
        feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats.cpu().numpy()

# -----------------------------
# SEARCH
# -----------------------------
def search_images(query):
    vec = encode_text(query)
    scores, indices = index.search(vec, TOP_K)

    return [
        {"path": image_paths[i], "score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="T-Shirt Semantic Search", layout="wide")
st.title("👕 T-Shirt Search (Text / Voice)")

mode = st.radio("Choose Input Method", ["Text Command", "Voice Command"])

query_text = None

# -----------------------------
# TEXT COMMAND
# -----------------------------
if mode == "Text Command":
    query_text = st.text_input("Describe the T-shirt")

# -----------------------------
# VOICE COMMAND (FIXED & WORKING)
# -----------------------------
elif mode == "Voice Command":
    st.write("🎙 Click the mic and speak")

    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=16000
    )

    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name

        st.audio(audio_bytes, format="audio/wav")

        segments, info = whisper_model.transcribe(audio_path)
        query_text = " ".join(seg.text for seg in segments)

        st.success("Recognized Text:")
        st.write(query_text)

        os.remove(audio_path)

# -----------------------------
# SHOW RESULTS
# -----------------------------
if query_text:
    results = search_images(query_text)

    st.subheader("✅ Best Match")
    best = results[0]
    st.image(best["path"], width=300)

    st.subheader("🔁 Recommended Similar T-Shirts")
    cols = st.columns(len(results) - 1)

    for col, item in zip(cols, results[1:]):
        col.image(item["path"], width=200)
