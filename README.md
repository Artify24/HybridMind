---
title: HybridMind
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: "1.45.1"
python_version: "3.10"
app_file: app.py
pinned: false
license: apache-2.0
short_description: PDF and web search Q&A with RAG
---

# HybridMind
HybridMind is a Streamlit app that answers questions using uploaded PDF documents and, when needed, supplements them with web search, Wikipedia, and Arxiv results.

Live app: https://hybridmind-wkby5wm5b2kq2rnubjyufo.streamlit.app/

## Features
- Upload one or more PDF files
- Build a local vector index from the uploaded documents
- Choose between three response modes automatically:
  - RAG for strong PDF matches
  - HYBRID for mixed PDF and external context
  - SEARCH for general web-based answers
- Keep chat history across turns in the same session
- Use Groq-hosted LLM responses with Hugging Face embeddings

## Tech Stack
- Streamlit
- LangChain
- Groq LLMs
- Hugging Face embeddings
- FAISS vector store
- DuckDuckGo Search
- Wikipedia and Arxiv tools
- PyPDF for document loading

## Prerequisites
- Python 3.10 or newer
- A Groq API key
- A Hugging Face token

## Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
\```bash
pip install -r requirements.txt
\```
4. Add a `.env` file in the project root with your keys:
\```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
\```

## Run Locally
Start the app with:
\```bash
streamlit run app.py
\```
Then open the local Streamlit URL shown in the terminal.

## How It Works
1. Upload PDF files.
2. The app splits the documents into chunks and stores them in FAISS.
3. When you ask a question, HybridMind checks the similarity score.
4. It selects the best mode:
   - RAG when the PDF context is highly relevant
   - HYBRID when external sources can help
   - SEARCH when the question is better answered from the web

## Project Structure
- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `temp/` - temporary folder used for uploaded PDF files

## Notes
- Uploaded PDFs are written to the `temp/` folder during processing.
- Make sure your API keys are valid before running the app.
