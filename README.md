# Predicting Breast Cancer Progression Stages Through Integrative Machine Learning and Explainable AI Approaches

## Overview

Breast cancer remains one of the leading causes of cancer-related mortality among women worldwide. Accurate prediction of cancer progression stages is essential for clinical decision-making, treatment planning, and patient prognosis.

This research presents an integrative Machine Learning (ML) and Explainable Artificial Intelligence (XAI) framework for multiclass breast cancer stage prediction. To improve robustness and generalizability, three publicly available datasets (Kaggle, METABRIC, and SEER) were merged into a unified dataset containing clinical and pathological features.

Multiple Machine Learning and Deep Learning models were evaluated, including Random Forest, XGBoost, LightGBM, CatBoost, SVM, Logistic Regression, MLP, CNN, and LSTM. Explainable AI techniques such as Feature Importance Analysis and SHAP (SHapley Additive exPlanations) were applied to improve model transparency and interpretability.

The proposed LightGBM–CatBoost Ensemble achieved the best performance with an accuracy of **98.36%**, demonstrating its effectiveness for multiclass breast cancer stage prediction.

---

## Research Presentation

This work was presented at:

**ECCT 2026 (Emerging Computing and Communication Technologies Conference 2026)**

---

## Authors

* Shammin Akter Thithi (First Author)
* Shahida Parvin Mini
* Fatema Tahsin Nowrin
* Muhaimen Khan (Corresponding Author)
* Khandaker Mohammad Mohi Uddin

Department of Computer Science and Engineering
Southeast University, Dhaka, Bangladesh

---

## Research Contributions

* Integration of Kaggle, METABRIC, and SEER breast cancer datasets.
* Comprehensive preprocessing and feature harmonization.
* Class imbalance handling using SMOTE.
* Feature selection using statistical and tree-based techniques.
* Comparative analysis of Machine Learning and Deep Learning models.
* Development of high-performing ensemble models.
* Explainable AI analysis using SHAP and Feature Importance.
* Multiclass breast cancer stage prediction (Stage 0–Stage 4).

---

## Dataset Information

### Data Sources

1. Kaggle Breast Cancer Prediction Dataset
2. METABRIC Breast Cancer Dataset
3. SEER Breast Cancer Dataset

### Final Dataset

* Total Records: 6,746
* Features: 13
* Target Variable: Cancer Stage
* Classes:

  * Stage 0
  * Stage 1
  * Stage 2
  * Stage 3
  * Stage 4

---

## Data Preprocessing

The following preprocessing steps were performed:

* Missing value imputation
* Duplicate removal
* Label Encoding
* Min-Max Normalization
* Feature Selection
* SMOTE-based class balancing

---

## Models Evaluated

### Machine Learning Models

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* LightGBM
* CatBoost
* AdaBoost
* Decision Tree
* Gradient Boosting
* Extra Trees Classifier
* K-Nearest Neighbors (KNN)

### Deep Learning Models

* Multilayer Perceptron (MLP)
* Convolutional Neural Network (CNN)
* Long Short-Term Memory (LSTM)

### Ensemble Models

* LightGBM + CatBoost
* LightGBM + CatBoost + Extra Trees
* CatBoost + Random Forest + Decision Tree
* CatBoost + Extra Trees + Decision Tree

---

## Best Model Performance

| Model               | Accuracy   |
| ------------------- | ---------- |
| LightGBM + CatBoost | **98.36%** |

### Additional Metrics

* Precision: 98.37%
* Recall: 98.37%
* F1-Score: 98.36%

---

## Explainable AI (XAI)

To improve model interpretability, the following XAI methods were employed:

* Feature Importance Analysis
* SHAP Summary Plot
* SHAP Bar Plot

Important features identified:

* Tumor Size
* NPI
* Lymph Nodes
* Age
* Survival Months

---

## Repository Structure

```text
├── data/
│   ├── kaggle_dataset.csv
│   ├── metabric_dataset.csv
│   └── seer_dataset.csv
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── feature_selection.ipynb
│   ├── model_training.ipynb
│   └── xai_analysis.ipynb
│
├── models/
│   ├── lightgbm_model.pkl
│   ├── catboost_model.pkl
│   └── ensemble_model.pkl
│
├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── shap_summary.png
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* CatBoost
* XGBoost
* TensorFlow / Keras
* SHAP
* Matplotlib
* Seaborn

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{thithi2026breastcancer,
  title={Predicting Breast Cancer Progression Stages Through Integrative Machine Learning and Explainable AI Approaches},
  author={Shammin Akter Thithi, Shahida Parvin Mini, Fatema Tahsin Nowrin, Muhaimen Khan and, Khandaker Mohammad Mohi Uddin},
  booktitle={ECCT 2026},
  year={2026}
}
```

---

## Disclaimer

This repository is intended for research and educational purposes only. It is not designed to replace professional medical diagnosis or clinical decision-making.
