# 🧠 DocuMind AI

**DocuMind AI** is an advanced RAG (Retrieval-Augmented Generation) application that allows users to chat with multiple PDF documents simultaneously. Built with Python, Streamlit, and Google Gemini.

## 🚀 Features
- **Multi-Document Analysis:** Upload and cross-reference multiple PDFs.
- **Smart Citations:** The bot cites the exact document and page number for every fact.
- **Hybrid Search:** Combines Vector Search (FAISS) with Gemini's General Knowledge.
- **Privacy First:** Documents are processed locally in memory.

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Custom Dark Mode UI)
- **LLM:** Google Gemini 2.0 Flash
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Framework:** LangChain

## 📦 How to Run

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/DocuMind-AI.git](https://github.com/YOUR_USERNAME/DocuMind-AI.git)
Install dependencies:

```bash

pip install -r requirements.txt
Configure your API Key:

Create a new file named .env in the project folder.

Add your Google API Key inside like this:

GOOGLE_API_KEY=your_actual_key_here
Run the app:

```bash

streamlit run app.py