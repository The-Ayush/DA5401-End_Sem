# DA5401 Project: Metric Score Prediction Using Regression (XGBoost + PCA)

**Name:** Ayush Kumar  
**Roll No:** DA24S019

---

## Overview

This project focuses on predicting evaluation metric scores using **regression models**, based on embeddings generated from metric names and text pairs.  
The evaluation metric for this task is **RMSE**, which makes it a **regression problem** rather than classification.

The pipeline includes:

1. **Embedding Extraction** – 768-dim metric embeddings + 768-dim pair-text embeddings.  
2. **Feature Construction** – Concatenation to form 1536-dimensional features.  
3. **Scaling & PCA** – Standardization followed by PCA (99% variance retained).  
4. **Resampling** – Handling imbalance in regression targets.  
5. **Hyperparameter Tuning** – Optuna optimization for best XGBoost params.  
6. **Final Training & Inference** – Training a full model and generating submissions.

---

## Files

- `train.csv` – Training dataset containing metric names and pair texts.  
- `test.csv` – Test dataset without target scores.  
- `metric_embeddings.npy` – Precomputed 768-dimensional metric embeddings.  
- `pair_embeddings.npy` – Precomputed 768-dimensional pair text embeddings.  
- `submission_gemma_6.csv` – Raw continuous predictions.  
- `submission_clipped_gemma_6.csv` – Rounded and clipped predictions (0–10).  
- `code.ipynb` or `train_script.py` – Entire training pipeline.  
- `README.md` – This file.

---

## Data Processing Steps

### **1. Loading Embeddings**
- Loaded **145 metric names**
- Loaded **768-dimensional embeddings**
- Created combined features:  
