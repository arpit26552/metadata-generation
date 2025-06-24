# app.py
import os
import json
import re
import tempfile
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

# Initialize KeyBERT model
kw_model = KeyBERT()

# Prompt template for chunk-level summarization
CHUNK_PROMPT = (
    "You are a smart assistant. Analyze the following content snippet and provide:\n"
    "- A short summary (1-2 lines)\n"
    "- Five important keywords (comma-separated)\n\n"
    "Content:\n"
    "{chunk}"
)

# Extract text (with OCR fallback for PDFs)
def extract_text(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    if ext == ".txt":
        return tmp_path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".docx":
        return "\n".join(p.text for p in Document(tmp_path).paragraphs)
    elif ext == ".pdf":
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t
                else:
                    img = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(img)
        return text
    raise ValueError("Unsupported file type")

# Clean up raw text
def clean_text(text: str) -> str:
    return " ".join(text.split())

# Split into chunks for summarization
def split_into_chunks(text: str, size=1700, overlap=50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Call Mistral API for summarization
def call_mistral(prompt: str, temperature=0.3) -> str:
    api_url = os.getenv("MISTRAL_API_URL")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_url or not api_key:
        raise ValueError("MISTRAL_API_URL and MISTRAL_API_KEY must be set as env vars")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "open-mistral-7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# Streamlit UI
st.set_page_config(page_title="Metadata & Summary Generator", layout="centered")

st.markdown('<h1 style="color:#4A90E2;text-align:center;">📄 Auto Metadata & Summary Generator</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("🔍 Processing your document..."):
        raw_text = extract_text(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = split_into_chunks(cleaned_text)

        partial_summaries = [call_mistral(CHUNK_PROMPT.format(chunk=c)) for c in chunks]
        combined = "\n\n".join(partial_summaries)

        # Final summarization and metadata prompt
        final_prompt = (
            "You are a document metadata expert. Use the provided chunk summaries to assemble structured information.\n\n"
            "Provide a result in JSON with the following keys:\n"
            "- title\n"
            "- author (or 'Not specified')\n"
            "- date ('Not specified')\n"
            "- keywords (list)\n"
            "- document_type (default to 'Article')\n"
            "- summary\n\n"
            f"Summaries:\n{combined}"
        )

        final_output = call_mistral(final_prompt)

    try:
        # Extract valid JSON from LLM response
        json_match = re.search(r'{[\s\S]+}', final_output)
        if not json_match:
            raise ValueError("⚠️ No valid JSON detected.")
        metadata = json.loads(json_match.group())

        # Enhance keywords using KeyBERT
        top_keywords = kw_model.extract_keywords(
            cleaned_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=10,
            use_maxsum=True,
            nr_candidates=20
        )
        metadata["keywords"] = [kw for kw, _ in top_keywords]

        # Display extracted metadata
        st.markdown('<h3 style="color:#1f77b4;">📌 <b>Extracted Metadata</b></h3>', unsafe_allow_html=True)
        st.json(metadata)

        # Show final summary nicely
        st.markdown('<h3 style="color:#2ca02c;">📝 <b>Wrapped Summary</b></h3>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:#333;font-size:16px;background:#f4f4f4;padding:15px;border-radius:8px'>{metadata['summary']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Download summary
        st.download_button(
            label="💾 Download Summary",
            data=metadata["summary"],
            file_name="summary.txt",
            mime="text/plain"
        )

        st.markdown("<hr><div style='text-align:center;color:#888'>Built by Arpit · Powered by Mistral AI</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Unable to process result: {e}")
