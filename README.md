# Hybrid IDS: Unsupervised Clustering & Explainable XGBoost

This repository contains the code for a hybrid machine learning pipeline that detects network intrusions (specifically DDoS attacks). It utilizes K-Means clustering for behavioral segmentation, SMOTE for class balancing, and an XGBoost ensemble for classification. SHAP is integrated for Explainable AI (XAI) insights.

## How to Reproduce
1. Install the required dependencies:
   `pip install -r requirements.txt`
2. Download the CIC-IDS-2017 dataset (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`) from Kaggle and place it in the root directory.
3. Run the main pipeline:
   `python ml_pipeline.py`