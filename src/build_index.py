# src/build_index.py

from index_builder import build_vectorstore

if __name__ == "__main__":
    build_vectorstore(persist=True)
