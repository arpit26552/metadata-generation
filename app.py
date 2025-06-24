# app.py
import os
import json
import textwrap
import tempfile
import re
from pathlib import Path
from typing import List
from collections import Counter

import streamlit as st
import requests
import pytesseract
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import pdfplumber
import string

# ───────────────────────────────
# Utility Functions
# ───────────────────────────────

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

def clean_text(text: str) -> str:
    return " ".join(text.split())

def segment_text(text: str, size=1700, overlap=50) -> List[str]:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)

def query_mistral(prompt: str, temperature=0.3) -> str:
    api_url = os.getenv("MISTRAL_API_URL")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_url or not api_key:
        raise ValueError("Please set MISTRAL_API_URL and MISTRAL_API_KEY environment variables.")

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

def extract_keywords_simple(text: str, top_n: int = 10) -> List[str]:
    words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
    stopwords = set(open("/usr/share/dict/words").read().split()) if os.path.exists("/usr/share/dict/words") else set()
    filtered = [w for w in words if len(w) > 3 and w not in stopwords]
    most_common = Counter(filtered).most_common(top_n)
    return [kw for kw, _ in most_common]

# ───────────────────────────────
# Streamlit App
# ───────────────────────────────

st.set_page_config(page_title="Metadata & Summary Generator", layout="centered")
st.markdown('<h1 style="color:#4A90E2;text-align:center;">📄 Auto Metadata & Summary Generator</h1>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("🔍 Analyzing your document..."):
        raw = extract_text(uploaded_file)
        cleaned = clean_text(raw)
        segments = segment_text(cleaned)

        prompt_template = (
            "You are an intelligent assistant. Read the following chunk and return:\n"
            "- A 1-2 sentence summary\n"
            "- Five keywords (comma-separated)\n\n"
            "Chunk:\n\"\"\"\n{chunk}\n\"\"\""
        )
        summaries = [query_mistral(prompt_template.format(chunk=part)) for part in segments]
        combined = "\n\n".join(summaries)

        final_prompt = (
            "You are a document metadata assistant. Read the summaries below and generate a single JSON output "
            "with the following fields: title, author, date, document_type (default 'Article'), summary, keywords (list).\n\n"
            "Summaries:\n"
            f"{combined}"
        )

        try:
            final_response = query_mistral(final_prompt)
            json_blocks = re.findall(r'\{.*?\}', final_response, re.DOTALL)
            metadata = json.loads(json_blocks[0]) if json_blocks else json.loads(final_response)

            metadata["keywords"] = extract_keywords_simple(cleaned, top_n=10)

            st.markdown('<h3 style="color:#1f77b4;">📌 <b>Extracted Metadata</b></h3>', unsafe_allow_html=True)
            st.json(metadata)

            st.markdown('<h3 style="color:#2ca02c;">📝 <b>Document Summary</b></h3>', unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:#333;font-size:16px;background:#f4f4f4;padding:15px;border-radius:8px'>{metadata['summary']}</div>",
                unsafe_allow_html=True
            )

            st.download_button(
                label="💾 Download Summary",
                data=metadata["summary"],
                file_name="summary.txt",
                mime="text/plain"
            )

            st.markdown("<hr><div style='text-align:center;color:#888'>Built by Arpit · Powered by Mistral AI</div>", unsafe_allow_html=True)

        except Exception as err:
            st.error(f"⚠️ Unable to process result: {err}")
