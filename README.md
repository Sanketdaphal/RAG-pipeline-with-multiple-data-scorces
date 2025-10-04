# 📚 RAG Pipeline with Multiple Data Sources

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** using **LangChain**, **Google Gemini**, and multiple external data sources such as **PDFs, Wikipedia, and Arxiv**.  
The app is deployed with **Streamlit** and provides an interactive chatbot interface.  
You can try it here
https://rag-pipeline-with-multiple-data-scorces-pblxdxaft8m8gnhc9ejwnz.streamlit.app/

---

## ✨ Features
- 📄 **PDF Loader** – Load and chunk documents (example: `SDE.pdf`).  
- 🔎 **Vector Store (FAISS)** – Store embeddings of documents for semantic search.  
- 🧠 **Embeddings (Gemini)** – Uses Google Generative AI embeddings (`models/gemini-embedding-001`).  
- 🌐 **External Tools**  
  - **Wikipedia**: Query and retrieve information from Wikipedia.  
  - **Arxiv**: Retrieve research papers from Arxiv.  
- 🤖 **Agent-based RAG** – Built using LangChain’s `create_tool_calling_agent`.  
- 🎛 **Streamlit UI** – User-friendly interface with feedback options.  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **LangChain** (`langchain`, `langchain-community`, `langchain-google-genai`)  
- **FAISS** (`faiss-cpu`)  
- **Streamlit**  
- **Google Gemini API** (via `ChatGoogleGenerativeAI`)  

---

## 📂 Project Structure


📂 rag-pipeline-with-multiple-data-sources
│── RAG.py                 # Main application script (Streamlit app)
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation
│── .env                   # API keys (not pushed to GitHub)
│
├── 📂 venv/               # Virtual environment (should be in .gitignore)
│
├── 📂 data/               # Store input files
│    └── SDE.pdf           # Example PDF used for RAG
│
├── 📂 vector_store/       # Saved FAISS or Chroma DB (optional)
│
└── 📂 .streamlit/         # Streamlit config (if needed)
     └── config.toml

