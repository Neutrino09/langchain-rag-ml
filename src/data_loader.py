# src/data_loader.py

import os
from config import DATA_DIR  # <- NO dot here


def load_texts_and_metadatas():
    """
    Load all .txt files from data/raw/ and return:
      - texts: list[str]
      - metadatas: list[dict]
    """
    texts = []
    metadatas = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            texts.append(text)
            metadatas.append({"source": filename})

    if not texts:
        raise ValueError(
            f"No .txt files found in {DATA_DIR}. "
            "Make sure machine_learning.txt is inside data/raw/"
        )

    return texts, metadatas


if __name__ == "__main__":
    print("Running data_loader.py ...")
    texts, metadatas = load_texts_and_metadatas()
    print(f"Loaded {len(texts)} text file(s).")
    print("First text snippet:\n")
    print(texts[0][:500])
    print("\nMetadata for first file:", metadatas[0])
