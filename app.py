# app.py
import os
import json
import textwrap
import tempfile
import re
from pathlib import Path
from typing import List

import streamlit as st
import requests
import pytesseract
from PIL import Image
from keybert import KeyBERT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import pdfplumber

# Initialize the keyword extractor
from sentence_transformers import SentenceTransformer
kw_model = KeyBERT(SentenceTransformer("all-MiniLM-L6-v2", device="cpu"))

# Prompt template used for individual content chunks
CHUNK_PROMPT = (
    "You are a smart assistant. Analyze the following content snippet and provide:\n"
    "- A short summary (1-2 lines)\n"
    "- Five important keywords (comma-separated)\n\n"
    "Content:\n"
    "{chunk}"
)

# Function to extract readable text from a file
def extract_text(uploaded_file) -> str:
    file_ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    if file_ext == ".txt":
        return tmp_path.read_text(encoding="utf-8", errors="ignore")
    elif file_ext == ".docx":
        return "\n".join(p.text for p in Document(tmp_path).paragraphs)
    elif file_ext == ".pdf":
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
                else:
                    img = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(img)
        return text
    raise ValueError(f"Unsupported file extension: {file_ext}")

# Normalize text spacing and line breaks
def clean_text(text: str) -> str:
    return " ".join(text.split())

# Break down text into manageable segments for LLM processing
def segment_text(text: str, size=1700, overlap=50) -> List[str]:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)

# Communicate with the Mistral API to get summarization output
def query_mistral(prompt: str, temperature=0.3) -> str:
    api_url = os.getenv("MISTRAL_API_URL")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_url or not api_key:
        raise ValueError("Environment variables for API not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "open-mistral-7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# ───────────────────────────────
# Streamlit App Interface
# ───────────────────────────────

st.set_page_config(page_title="Metadata & Summary Generator", layout="centered")

st.markdown('<h1 style="color:#4A90E2;text-align:center;">📄 Auto Metadata & Summary Generator</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("🔍 Analyzing your document..."):
        raw = extract_text(uploaded_file)
        cleaned = clean_text(raw)
        segments = segment_text(cleaned)

        summaries = [query_mistral(CHUNK_PROMPT.format(chunk=part)) for part in segments]
        combined_summary = "\n\n".join(summaries)

        final_instruction = (
            "You are a document metadata expert. Use the provided chunk summaries to assemble structured information.\n\n"
            "Provide a result in JSON with the following keys:\n"
            "- title\n"
            "- author (or 'Not specified')\n"
            "- date ('Not specified')\n"
            "- keywords (list)\n"
            "- document_type (default to 'Article')\n"
            "- summary\n\n"
            f"Summaries:\n{combined_summary}"
        )

        response = query_mistral(final_instruction)

        try:
            json_blocks = re.findall(r'\{.*?\}', response, re.DOTALL)
            metadata = json.loads(json_blocks[0]) if json_blocks else json.loads(response)

            keyphrases = kw_model.extract_keywords(
                cleaned,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=10,
                use_maxsum=True,
                nr_candidates=20
            )
            metadata["keywords"] = [kw for kw, _ in keyphrases]

            st.markdown('<h3 style="color:#1f77b4;">📌 <b>Extracted Metadata</b></h3>', unsafe_allow_html=True)
            st.json(metadata)

            st.markdown('<h3 style="color:#2ca02c;">📝 <b>Wrapped Summary</b></h3>', unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:#333;font-size:16px;background:#f4f4f4;padding:15px;border-radius:8px'>{metadata['summary']}</div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            st.download_button(
                label="💾 Download Summary",
                data=metadata["summary"],
                file_name="summary.txt",
                mime="text/plain"
            )

            st.markdown("<hr><div style='text-align:center;color:#888'>Built by Arpit · Powered by Mistral AI</div>", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"⚠️ Unable to process result: {err}")
