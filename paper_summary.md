# Paper Summary

**Title**: BERT Based Sarcasm Detection Hybrid Approach For Hinglish Languages Using GloVe Embedding

**Authors**: Ayush, Ritika Choudhary, Kishan Kumar  
**Institution**: Galgotias University, Greater Noida, India

---

## Problem

Detecting sarcasm in **Hinglish** (Hindi-English code-mixed) social media text is difficult because:
- Code-switching creates mixed lexical and informal spelling patterns
- Annotated Hinglish data is scarce
- Standard NLP pipelines are ineffective on code-mixed text
- Deep models lack interpretability

Prior best results: ~78.5% accuracy (Aggarwal et al., BiLSTM) and ~76% F1 (Sharma et al., fastText + Naïve Bayes).

---

## Proposed Solution

A **hybrid feature fusion** system with three parallel channels:

1. **Contextual embeddings** — XLM-RoBERTa base → BiLSTM (bidirectional, 256 hidden) → Multi-Head Self-Attention (4 heads)
2. **Lexical features** — TF-IDF word n-grams (1–2) + character n-grams (3–5)
3. **Emotion encoding** — Sentiment polarity score per word from a Hinglish emotion lexicon

All three are concatenated and passed to a **weighted ensemble** of SVM, Logistic Regression, and the neural hybrid model. Model weights are proportional to validation accuracy.

---

## Key Findings

- The ensemble (~95.5%) outperforms all three single models
- Emotion encoding is especially helpful on borderline cases
- Attention heatmaps highlight semantically loaded words (positive adjective in negative context)
- TF-IDF features provide interpretability that pure deep models lack

---

## Future Directions

- Multimodal cues (emojis, images, formatting)
- Few-shot and continual learning with DeBERTa-v3 / XLM-RoBERTa-large
- Extension to other code-mixed languages (e.g., Spanglish, Taglish)
- Cross-domain generalization (chat conversations, forum posts)
- Deeper attention-to-linguistic-theory interpretability mapping
