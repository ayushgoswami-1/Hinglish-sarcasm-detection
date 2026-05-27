# Results

## Classification Performance

Results on the held-out test set (1,919 samples):

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Not Sarcastic (0) | 0.95 | 0.94 | 0.95 | 817 |
| Sarcastic (1) | 0.95 | 0.97 | 0.96 | 1,102 |
| **Accuracy** | | | **0.95** | 1,919 |
| Macro Avg | 0.95 | 0.95 | 0.95 | 1,919 |
| Weighted Avg | 0.95 | 0.95 | 0.95 | 1,919 |

## Confusion Matrix

|  | Predicted: Not Sarcastic | Predicted: Sarcastic |
|---|---|---|
| **Actual: Not Sarcastic** | 766 (TN) | 51 (FP) |
| **Actual: Sarcastic** | 37 (FN) | 1065 (TP) |

## Model Comparison

| Model | Accuracy |
|---|---|
| Logistic Regression | 90.4% |
| Linear SVM | 91.6% |
| XLM-R + BiLSTM + Attention | 94.0% |
| **Weighted Ensemble (Proposed)** | **~95.5%** |

## Comparison with Prior Work

| Model | Accuracy |
|---|---|
| **Proposed Model** | **95.5%** |
| Deep Learning (2025) | 94.0% |
| Transformer (2024) | 93.0% |
| Fuzzy Hybrid (2023) | 89.0% |
| Autoencoder Hybrid (2022) | 88.0% |
| Hinglish Transformer Benchmark (2026) | 84.0% |
| Aggarwal et al. BiLSTM (baseline) | 78.5% |

## Curves

After training, the notebook generates:
- `accuracy_vs_epoch.png` — Training accuracy over 5 epochs
- `loss_vs_epoch.png` — Training loss over 5 epochs  
- `confusion_matrix.png` — Neural model confusion matrix
- `roc_curve.png` — ROC curve (AUC > 0.97)
- `precision_recall_curve.png` — Precision-Recall curve (AP = 0.99)

Run the notebook to regenerate these plots.
