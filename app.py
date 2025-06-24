
import os
import json
import textwrap
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
import requests
from keybert import KeyBERT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from docx import Document
import pdfplumber

# Load KeyBERT model once
kw_model = KeyBERT()

# Prompt used per chunk
PROMPT_TEMPLATE = """
You are an intelligent assistant. Read this content chunk and return:
- A 1-2 sentence summary
- Five keywords (comma-separated)

Chunk:
\"\"\"{content_chunk}\"\"\"
"""

# ───────────────────────────────
# Utility Functions
# ───────────────────────────────

def extract_text_from_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    if suffix == ".txt":
        return tmp_path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".docx":
        doc = Document(tmp_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif suffix == ".pdf":
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    raise ValueError(f"Unsupported file type: {suffix}")

def preprocess_text(text: str) -> str:
    return " ".join(text.split())

def split_text_into_chunks(text: str, size: int = 1700, overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_text(text)

def call_mistral(prompt: str, temperature: float = 0.3) -> str:
    api_url = os.getenv("MISTRAL_API_URL")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_url or not api_key:
        raise ValueError("Please set MISTRAL_API_URL and MISTRAL_API_KEY env vars")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "open-mistral-7b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    resp = requests.post(api_url, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ───────────────────────────────
# Streamlit UI
# ───────────────────────────────

st.set_page_config(page_title="Metadata & Summary Generator", layout="centered")

st.markdown(
    '<h1 style="color:#4A90E2;text-align:center;">📄 Auto Metadata & Summary Generator</h1>',
    unsafe_allow_html=True
)

file = st.file_uploader("📂 Upload PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if file:
    with st.spinner("⏳ Processing the uploaded file..."):
        raw_text = extract_text_from_file(file)
        clean_text = preprocess_text(raw_text)
        chunks = split_text_into_chunks(clean_text)

        summaries = [call_mistral(PROMPT_TEMPLATE.format(content_chunk=chunk)) for chunk in chunks]
        combined = "\n\n".join(summaries)

        # Final summarization prompt
        final_prompt = f"""
You are a smart metadata assistant. Below are partial summaries of a document generated from different chunks.

Your task is to first read the chunks carefully to combine them into a **single, coherent metadata JSON object** with meaningful values.

Infer the **title** and **author** based on the document as a whole, even if not explicitly mentioned.
Generate a meaningful, concise **summary** for the full document.
Merge and deduplicate the keywords intelligently.
Assume the document type is "Article" unless there's a clear reason to choose otherwise.

Return the result in this JSON format:
{{
  "title": "Meaningful title of the whole document",
  "author": "Author name (or 'Not specified' if not found)",
  "date": "Not specified",
  "keywords": ["keyword1", "keyword2", "..."],
  "document_type": "Article",
  "summary": "Clean, concise summary of the full document."
}}

Here are the partial summaries:
\"\"\"{combined}\"\"\"
"""
        final_output = call_mistral(final_prompt)

    try:
        parsed = json.loads(final_output)

        # Optional: Use KeyBERT for better keywords
        kb_keywords = kw_model.extract_keywords(
            clean_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=10,
            use_maxsum=True,
            nr_candidates=20
        )
        parsed["keywords"] = [kw for kw, _ in kb_keywords]

        # ── Styled Output ──
        st.markdown('<h3 style="color:#1f77b4;">📌 <b>Extracted Metadata</b></h3>', unsafe_allow_html=True)
        st.json(parsed)

        st.markdown('<h3 style="color:#2ca02c;">📝 <b>Wrapped Summary</b></h3>', unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:#333333; font-size:16px; line-height:1.6; background-color:#f4f4f4; padding:15px; border-radius:8px'>{parsed['summary']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.download_button(
            label="💾 Download Summary",
            data=parsed["summary"],
            file_name="summary.txt",
            mime="text/plain"
        )

        st.markdown("<hr><div style='text-align:center;color:#888'>Made with ❤️ by Arpit · Powered by Mistral AI</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Failed to parse output: {e}")
