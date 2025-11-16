# src/index_builder.py

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    INDEX_DIR,
    OPENAI_API_KEY,
)
from data_loader import load_texts_and_metadatas


def build_vectorstore(persist: bool = True) -> FAISS:
    """
    Load raw texts, split into chunks, create embeddings,
    and build a FAISS vector store. Optionally save to disk.
    """
    print("Loading texts from data/raw ...")
    texts, metadatas = load_texts_and_metadatas()
    print(f"Loaded {len(texts)} text file(s).")

    print("Splitting texts into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    all_chunk_metadatas = []

    # Split each text individually so source metadata stays correct
    for text, meta in zip(texts, metadatas):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_chunk_metadatas.extend([meta] * len(chunks))

    print(f"Created {len(all_chunks)} chunks.")

    print("Creating embeddings and building FAISS index...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_chunk_metadatas,
    )

    if persist:
        os.makedirs(INDEX_DIR, exist_ok=True)
        print(f"Saving FAISS index to {INDEX_DIR} ...")
        vectorstore.save_local(INDEX_DIR)

    print("Vector store built successfully.")
    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Load a previously saved FAISS vector store from disk.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


if __name__ == "__main__":
    build_vectorstore(persist=True)
