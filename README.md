

A clean, modular Retrieval-Augmented Generation (RAG) application built using **LangChain**, **OpenAI embeddings**, **FAISS**, and a **Streamlit chat UI**.
It answers questions about Machine Learning using a custom knowledge base extracted from the Machine Learning Wikipedia page.

This project was built as part of the **Generative AI Engineer Intern Assignment** for **Pythrust Technologies**.

---

## **📌 Features**

* **Custom Knowledge Base**
  Ingests the Machine Learning Wikipedia page (stored locally as `.txt`).

* **Document Processing Pipeline**

  * Text loading
  * Chunking (Recursive Character Splitter)
  * Embedding generation using **OpenAI text-embedding-3-small**
  * Vector store creation with **FAISS**

* **Retrieval-Augmented Generation (RAG)**

  * FAISS retriever
  * RetrievalQA chain (LangChain)
  * `ChatOpenAI` model for grounded response generation
  * Source citation for each answer

* **Streamlit UI**

  * Clean, modern, production-style chat interface
  * Conversation memory
  * Source tags for transparency
  * Side panel with app info and usage instructions

* **Modular Codebase**
  Each responsibility (config, indexing, retrieval, UI) is separated into dedicated modules.
  <img width="1470" height="956" alt="Screenshot 2025-11-16 at 6 10 34 PM" src="https://github.com/user-attachments/assets/882618e3-7d86-4add-a8e9-5a1941e21a78" />


---

## **📁 Project Structure**

```
langchain-rag-ml/
│
├── data/
│   └── raw/
│       └── machine_learning.txt        ← Knowledge base
│
├── artifacts/
│   └── faiss_index/                    ← Auto-generated FAISS index
│
├── src/
│   ├── app_streamlit.py                ← Streamlit chat UI
│   ├── config.py                       ← Env-based config (OpenAI key, paths, settings)
│   ├── data_loader.py                  ← Loads text files from data/raw/
│   ├── index_builder.py                ← Chunking, embedding, FAISS index creation
│   ├── rag_chain.py                    ← RAG pipeline + ask_question() API
│   ├── build_index.py                  ← Script to build the FAISS index
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

## **⚙️ Installation**

Clone the repository:

```bash
git clone https://github.com/Neutrino09/langchain-rag-ml.git
cd langchain-rag-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **🔑 Set OpenAI API Key**

The app reads your API key from an **environment variable**, so set it like this:

### macOS / Linux:

```bash
export OPENAI_API_KEY="your-key-here"
```

### Windows (Powershell):

```powershell
setx OPENAI_API_KEY "your-key-here"
```

> Do **not** hardcode your key into the config file if pushing to GitHub.

---

## **📚 Step 1 — Prepare the Dataset**

Place your source documents in:

```
data/raw/
```

By default, the repo comes with:

```
machine_learning.txt
```

You can add more `.txt` files — the pipeline will ingest all of them automatically.

---

## **🧱 Step 2 — Build the FAISS Index**

Before running the app, embed your documents and create the vector index:

```bash
python src/build_index.py
```

This generates:

```
artifacts/faiss_index/
```

---

## **💬 Step 3 — Run the Streamlit App**

```bash
streamlit run src/app_streamlit.py
```

Then open:

```
http://localhost:8501
```

You can now chat with the RAG system.

---

## **🧠 How It Works**

1. **Ingestion**
   `data_loader.py` loads all `.txt` files under `data/raw/`.

2. **Chunking & Embeddings**
   `index_builder.py` splits documents, generates embeddings (OpenAI), and builds FAISS index.

3. **Retrieval**
   `rag_chain.py` loads the FAISS index and retrieves the top-k relevant chunks for each query.

4. **Generation**
   A `RetrievalQA` chain calls the LLM (`gpt-4o-mini`) with the retrieved context.

5. **UI**
   `app_streamlit.py` provides a chat interface and displays source citations.

---

## **📝 Notes**

* The entire pipeline is modular and easy to extend:

  * Swap in a different LLM
  * Add more documents
  * Change vector store or retriever
  * Integrate a proper memory module

* The FAISS index is not stored in git; it’s generated locally.

---

