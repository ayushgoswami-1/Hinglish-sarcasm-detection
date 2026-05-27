# -*- coding: utf-8 -*-
"""
predict.py — Standalone sarcasm prediction for Hinglish text.

Usage:
    python src/predict.py --text "Wow great job failing again"
    python src/predict.py --file input_texts.txt

Requires a trained model checkpoint saved via train.py.
For first-time use, run the full notebook to train and save the model.
"""

import argparse
import re
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from langdetect import detect


# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Hybrid Model Definition
# ─────────────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained("xlm-roberta-base")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(512, 4, batch_first=True)
        self.emotion_fc = nn.Linear(1, 32)
        self.fc = nn.Linear(512 + 32, 1)

    def forward(self, input_ids, attention_mask, emotion):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = torch.mean(x, dim=1)
        emotion = torch.relu(self.emotion_fc(emotion.unsqueeze(1)))
        x = torch.cat((x, emotion), dim=1)
        return torch.sigmoid(self.fc(x))


# ─────────────────────────────────────────────
# Load Saved Artifacts
# ─────────────────────────────────────────────
def load_artifacts(checkpoint_dir: str = "checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridModel().to(device)
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "hybrid_model.pt"), map_location=device)
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Load sklearn models and TF-IDF vectorizers
    with open(os.path.join(checkpoint_dir, "artifacts.pkl"), "rb") as f:
        artifacts = pickle.load(f)

    return model, tokenizer, artifacts, device


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(text: str, model, tokenizer, artifacts, device) -> dict:
    text = preprocess(text)
    try:
        language = detect(text)
    except Exception:
        language = "unknown"

    tfidf_word = artifacts["tfidf_word"]
    tfidf_char = artifacts["tfidf_char"]
    svm = artifacts["svm"]
    lr = artifacts["lr"]
    emotion_dict = artifacts["emotion_dict"]
    w_svm = artifacts["w_svm"]
    w_lr = artifacts["w_lr"]
    w_nn = artifacts["w_nn"]

    # TF-IDF
    word_vec = tfidf_word.transform([text]).toarray()
    char_vec = tfidf_char.transform([text]).toarray()
    tfidf_vec = np.hstack((word_vec, char_vec))

    svm_pred = svm.predict(tfidf_vec)[0]
    lr_pred = lr.predict(tfidf_vec)[0]

    # Neural model
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    emotion_score = sum(emotion_dict.get(w, 0) for w in text.split())
    emotion_tensor = torch.tensor([emotion_score], dtype=torch.float).to(device)

    with torch.no_grad():
        nn_prob = model(enc["input_ids"], enc["attention_mask"], emotion_tensor).item()

    nn_pred = 1 if nn_prob > 0.5 else 0
    final_score = w_svm * svm_pred + w_lr * lr_pred + w_nn * nn_pred
    final = 1 if final_score >= 0.5 else 0

    return {
        "text": text,
        "language": language,
        "is_sarcastic": bool(final),
        "label": "Sarcastic" if final else "Not Sarcastic",
        "confidence": round(final_score * 100, 2),
        "nn_prob": round(nn_prob * 100, 2),
        "svm_pred": int(svm_pred),
        "lr_pred": int(lr_pred),
    }


def print_report(result: dict):
    print("\n🔍 INPUT ANALYSIS REPORT")
    print("─" * 50)
    print(f"Text        : {result['text']}")
    print(f"Language    : {result['language']}")
    print(f"Final Class : {'✅ Sarcastic' if result['is_sarcastic'] else '❌ Not Sarcastic'}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"  ↳ Neural  : {result['nn_prob']}%")
    print(f"  ↳ SVM     : {'Sarcastic' if result['svm_pred'] else 'Not Sarcastic'}")
    print(f"  ↳ LR      : {'Sarcastic' if result['lr_pred'] else 'Not Sarcastic'}")
    print("─" * 50)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hinglish Sarcasm Detector")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="Path to a .txt file (one sentence per line)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory with saved model artifacts")
    args = parser.parse_args()

    print("Loading model artifacts...")
    model, tokenizer, artifacts, device = load_artifacts(args.checkpoint_dir)
    print(f"Model loaded. Running on: {device}\n")

    if args.text:
        result = predict(args.text, model, tokenizer, artifacts, device)
        print_report(result)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            result = predict(line, model, tokenizer, artifacts, device)
            print_report(result)

    else:
        # Interactive mode
        print("Interactive mode — type a sentence and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                text = input("Enter text: ")
                if text.strip():
                    result = predict(text, model, tokenizer, artifacts, device)
                    print_report(result)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
