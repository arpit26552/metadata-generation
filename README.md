# 📄 Auto Metadata & Summary Generator

An intelligent Streamlit-based app that extracts structured **metadata** and a **concise summary** from uploaded documents (PDF, DOCX, or TXT). Powered by Mistral AI and enhanced with KeyBERT for high-quality keyword extraction.

🔗 **Live App**: [Try it here](https://metadata-generation-ebtysf3m28gnrrusfqvwak.streamlit.app/)

---

## 🚀 Features

- 📂 Upload documents: PDF, Word (DOCX), or plain text
- 🧠 Chunk-wise summarization using LLM (Mistral)
- 🔍 Metadata extraction:
  - Title
  - Author
  - Date
  - Document type
  - Top keywords
- 📝 Final wrapped summary with clean formatting
- 💾 Download the summary with one click

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io) – UI framework
- [Mistral API](https://mistral.ai) – LLM-powered chunk summarization
- [KeyBERT](https://github.com/MaartenGr/KeyBERT) – Keyword extraction
- [pdfplumber](https://github.com/jsvine/pdfplumber) – PDF parsing
- [python-docx](https://github.com/python-openxml/python-docx) – DOCX parsing
- [LangChain](https://www.langchain.com/) – Chunk splitter

---

## 🛠️ How to Run Locally

1. Clone this repo:
git clone [clone](https://github.com/arpit26552/metadata-generation.git)
cd metadata-generation

2. Install requirements
Install dependencies: pip install -r requirements.txt

3. Setup API Keys
  - [Streamlit](https://streamlit.io) – UI framework
  - [Mistral API](https://mistral.ai) – LLM-powered chunk summarization
  - [KeyBERT](https://github.com/MaartenGr/KeyBERT) – Keyword extraction
  - [pdfplumber](https://github.com/jsvine/pdfplumber) – PDF parsing
  - [python-docx](https://github.com/python-openxml/python-docx) – DOCX parsing
  - [LangChain](https://www.langchain.com/) – Chunk splitter

 4. Run the Streamlit app
  - streamlit run app.py


✍️ Author
Made with ❤️ by Arpit Kumar
