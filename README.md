# DA5401 Project: Metric Learning for Conversational AI Evaluation

**Name:** Ayush Kumar  
**Roll No:** DA24S019  

---

## Overview

This project focuses on predicting the *fitness score* (1–10) between an **evaluation metric definition** and a **prompt–response conversation pair**.  
The task is treated as a **regression problem** because the evaluation score is continuous and evaluated using RMSE.

We implemented the following major steps:

1. **Text Embedding using EmbeddingGemma** – Encoded conversation text into 768-dimensional vectors.  
2. **Metric Embedding Fusion** – Combined metric embeddings with conversation embeddings (total 1536 features).  
3. **Dimensionality Reduction (PCA)** – Reduced feature space while retaining 99% variance.  
4. **Custom Sampling Strategy** – Balanced the highly imbalanced score distribution using controlled undersampling.  
5. **XGBoost Regression (Optuna-Tuned)** – Final model selected using 5-fold CV to minimize RMSE.

This pipeline produced the best stability and predictive performance after experimenting with multiple sampling and modeling strategies.

---

## Files

- `train_data.json` – Training dataset containing prompts, responses, metric names, and scores.  
- `test_data.json` – Test dataset without labels.  
- `metric_names.json` – List of metric names.  
- `metric_name_embeddings.npy` – 768-dimensional metric embeddings.  
- `submission_gemma_6.csv` – Raw predicted scores.  
- `submission_clipped_gemma_6.csv` – Rounded and clipped scores (0–10).  
- `code.ipynb` – Notebook containing the full embedding, PCA, sampling, and XGBoost pipeline.  
- `README.md` – This file.

---

## Instructions

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost optuna sentence-transformers
