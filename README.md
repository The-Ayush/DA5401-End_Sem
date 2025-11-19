# DA5401 Project: Metric Score Prediction

**Name:** Ayush Kumar  
**Roll No:** DA24S019  

---

## Project Summary

This project focuses on predicting evaluation metric scores (1–10) using a **regression-based pipeline**.  
Metric embeddings and conversation embeddings were combined to form numerical feature vectors, scaled, and reduced using PCA.  
Class imbalance was handled through custom resampling, and **XGBoost Regression** (tuned via Optuna) was used as the final model.

---

## Workflow Overview

- Loaded metric names and 768-dim embeddings  
- Generated conversation embeddings using EmbeddingGemma  
- Combined text + metric embeddings → 1536-dim features  
- Applied StandardScaler + PCA (retaining 99% variance)  
- Performed undersampling/oversampling to handle imbalance  
- Tuned and trained XGBoost using Optuna best parameters  
- Generated raw and rounded/clipped submissions  

---

## Files Included

- Training & test JSON datasets  
- Metric embeddings  
- Final submissions (`submission_gemma.csv` and clipped version)  
- Notebook / script with full pipeline  
- README summarizing methodology  

---

## Key Notes

- RMSE metric → Regression was the correct modeling choice  
- PCA significantly reduced dimensionality  
- Resampling improved score distribution for training  
- XGBoost performed best after tuning  

---
