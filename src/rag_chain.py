# src/rag_chain.py

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from config import LLM_MODEL, OPENAI_API_KEY, TOP_K
from index_builder import load_vectorstore


def get_retriever():
    """
    Load the FAISS vectorstore and return a retriever.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    return retriever


def get_qa_chain():
    """
    Build a RetrievalQA chain using ChatOpenAI and the FAISS retriever.
    """
    retriever = get_retriever()

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.0,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


def ask_question(question: str) -> Dict[str, Any]:
    """
    Run the QA chain on a question and return:
      - answer text
      - list of source filenames
    """
    qa_chain = get_qa_chain()
    result = qa_chain({"query": question})

    answer = result["result"]
    source_docs = result.get("source_documents", [])

    sources = []
    for d in source_docs:
        src = d.metadata.get("source", "unknown")
        sources.append(src)

    sources = sorted(list(set(sources)))

    return {
        "answer": answer,
        "sources": sources,
    }


if __name__ == "__main__":
    q = "What is machine learning?"
    out = ask_question(q)
    print("Q:", q)
    print("\nAnswer:\n", out["answer"])
    print("\nSources:", out["sources"])
