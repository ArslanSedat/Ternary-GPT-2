"""
Prepare Moliere's L'Avare dataset imported in fine from my Hugging Face (check README)
"""

import argparse
import pickle
import re
from pathlib import Path
import numpy as np
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent

def load_moliere_dataset(repo_id, split="train", text_column="text"):
    dataset = load_dataset(repo_id, split=split)
    texts = [value for value in dataset[text_column] if isinstance(value, str) and value.strip()]
    text = "\n\n".join(texts)
    return text

def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip() + "\n"

def build_vocabulary(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, stoi, itos

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def split_dataset(text, train_ratio=0.9):
    n = len(text)
    split_index = int(n * train_ratio)
    train_text = text[:split_index]
    val_text = text[split_index:]
    return train_text, val_text

def prepare_dataset(text, train_ratio=0.9):
    input_path = DATA_DIR / "input.txt"
    input_path.write_text(text, encoding="utf-8")
    vocab_size, stoi, itos = build_vocabulary(text)
    train_text, val_text = split_dataset(text, train_ratio=train_ratio)
    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(DATA_DIR / "train.bin")
    val_ids.tofile(DATA_DIR / "val.bin")
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(DATA_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print("files generated")

def main():
    parser = argparse.ArgumentParser(description="Moliere dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset ID")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    text = load_moliere_dataset(repo_id=args.repo_id, split=args.split,
                                text_column=args.text_column)
    text = clean_text(text)
    prepare_dataset(text, train_ratio=args.train_ratio)

if __name__ == "__main__":
    main()