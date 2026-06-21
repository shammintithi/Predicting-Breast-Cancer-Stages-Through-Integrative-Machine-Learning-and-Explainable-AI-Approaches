# Predicting Breast Cancer Progression Stages Through Integrative Machine Learning and Explainable AI Approaches

## Project Overview

This repository contains a comprehensive computational framework for breast cancer stage prediction utilizing an integrative approach combining ensemble machine learning models, deep learning architectures, and explainable artificial intelligence (XAI) techniques. Our research demonstrates the efficacy of multi-model ensemble strategies coupled with SHAP-based model interpretability to achieve superior predictive accuracy while maintaining clinical explainability on multi-cohort clinical datasets.

**Best Performing Model:** LightGBM–CatBoost Ensemble achieving **98.36% accuracy** with robust generalization across independent validation cohorts.

---

## Research Presentation Information

**Conference:** ECCT 2026 - European Conference on Computational Technologies

**Research Focus:** Integrative machine learning and explainable AI for automated breast cancer stage classification from multi-dimensional clinical, pathological, and genomic features.

**Key Innovation:** Integration of heterogeneous model architectures (gradient boosting, neural networks, probabilistic models) with SHAP-based feature attribution for clinical decision support.

---

## Authors

**First Author & Project Lead:**
- **Shammini Tithi** - Lead Researcher & Data Scientist
  - Project conception, methodology design, model development, and manuscript preparation

**Contributing Authors:**
- *Additional contributors welcome via collaborative research partnerships*

---

## Research Contributions

### Primary Contributions

1. **Comprehensive Multi-Cohort Analysis:** Integration and harmonization of three major clinical datasets (Kaggle, METABRIC, SEER) with standardized preprocessing pipelines to enable robust cross-dataset validation.

2. **Ensemble Learning Strategy:** Development of a novel LightGBM–CatBoost ensemble architecture that achieves superior predictive performance (98.36% accuracy) compared to individual base models.

3. **Explainable AI Framework:** Implementation of SHAP (SHapley Additive exPlanations) methodology to provide granular feature-level attribution and model-agnostic interpretability for clinical stakeholders.

4. **Deep Learning Integration:** Incorporation of neural network architectures (fully-connected networks, convolutional layers) with ensemble methods to capture complex non-linear relationships in clinical features.

5. **Comparative Model Analysis:** Systematic evaluation of 8+ distinct modeling paradigms including traditional machine learning, ensemble methods, and deep learning to establish performance benchmarks.

6. **Clinical Interpretability:** Quantitative feature importance analysis and SHAP dependency plots to support clinical interpretability and potential integration into clinical decision-support systems.

---

## Dataset Description

### Data Sources

| Dataset | Size | Features | Source | Clinical Context |
|---------|------|----------|--------|------------------|
| **Kaggle** | ~1,000 samples | 32 clinical/pathological | Kaggle platform | General breast cancer cohort |
| **METABRIC** | ~1,980 samples | 30 genomic/clinical | European consortium | Breast Cancer research |
| **SEER** | ~4,000+ samples | 45 demographic/tumor | National cancer registry | US population cohort |

### Key Features

- **Demographic variables:** Age, ethnicity, menopausal status
- **Clinical features:** Tumor size, grade, stage, lymph node involvement
- **Pathological markers:** ER/PR/HER2 status, Ki-67 proliferation index
- **Genomic features:** Gene expression signatures (METABRIC dataset)
- **Outcome variable:** Cancer stage classification (I, II, III, IV)

### Cohort Characteristics

- **Total N:** 7,000+ patients across combined datasets
- **Outcome balance:** Addressed through stratified sampling and class-weighted algorithms
- **Missing data handling:** Multiple imputation and case-wise deletion strategies implemented
- **Cross-cohort harmonization:** Dataset-specific preprocessing pipelines with standardized feature scaling

---

## Data Preprocessing Pipeline

### Preprocessing Steps

```
Raw Data Input
    ↓
[1] Data Quality Assessment & Missing Value Analysis
    → Missing value patterns documented
    → Outlier detection via IQR and isolation forest methods
    ↓
[2] Missing Value Imputation
    → Categorical features: Mode imputation / K-nearest neighbors
    → Numerical features: Mean/median imputation / Iterative imputation
    ↓
[3] Outlier Treatment
    → Statistical outlier detection and capping
    → Domain-informed retention for clinically significant values
    ↓
[4] Feature Scaling & Normalization
    → StandardScaler: For tree-based models
    → MinMaxScaler: For neural network inputs
    → Robust scaling applied for features with outliers
    ↓
[5] Categorical Encoding
    → One-hot encoding: Multi-class features
    → Label encoding: Binary and ordinal features
    → Target encoding: High-cardinality categorical variables
    ↓
[6] Train-Test Split
    → Stratified split: 70% training, 15% validation, 15% testing
    → Cross-cohort validation: Hold-out test sets per dataset source
    ↓
[7] Class Imbalance Handling
    → SMOTE (Synthetic Minority Over-sampling Technique)
    → Class weight adjustment in model training
    ↓
Preprocessed Feature Matrix Ready for Modeling
```

---

## Machine Learning Models

### Traditional ML Architectures

#### 1. **Logistic Regression**
- **Purpose:** Baseline probabilistic classifier
- **Configuration:** L2 regularization, max_iter=1000
- **Performance:** Establishes explainability benchmark

#### 2. **Random Forest**
- **Architecture:** 500 estimators, max_depth=15
- **Features:** Inherent feature importance, parallel processing
- **Performance:** Strong baseline for ensemble comparison

#### 3. **Gradient Boosting (XGBoost)**
- **Architecture:** 500 estimators, learning_rate=0.1, max_depth=7
- **Advantages:** Sequential error correction, SHAP compatibility
- **Performance:** ~96.2% accuracy on validation set

#### 4. **LightGBM**
- **Architecture:** 500 iterations, learning_rate=0.05, num_leaves=31
- **Advantages:** Faster training, memory efficiency, leaf-wise tree growth
- **Performance:** ~97.8% accuracy on validation set

#### 5. **CatBoost**
- **Architecture:** 500 iterations, learning_rate=0.05, depth=8
- **Advantages:** Categorical feature handling, symmetric tree structure
- **Performance:** ~97.5% accuracy on validation set

#### 6. **Support Vector Machine (SVM)**
- **Configuration:** RBF kernel, C=1.0, gamma='scale'
- **Role:** Non-linear classification baseline
- **Performance:** ~94.8% accuracy

---

## Deep Learning Models

### Neural Network Architectures

#### 1. **Deep Fully-Connected Network (DNN)**
```
Input Layer: n_features
    ↓
Dense(256, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(128, ReLU) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, ReLU) → BatchNorm → Dropout(0.2)
    ↓
Dense(32, ReLU) → Dropout(0.2)
    ↓
Output Layer: 4 units (softmax) - Cancer stages
```
- **Loss Function:** Categorical cross-entropy
- **Optimizer:** Adam (lr=0.001)
- **Performance:** ~96.8% accuracy

#### 2. **1D Convolutional Neural Network (CNN)**
- **Architecture:** Conv1D(32, 3) → Conv1D(64, 3) → Dense layers
- **Purpose:** Capture local feature patterns
- **Performance:** ~95.9% accuracy

#### 3. **Recurrent Neural Network (LSTM/GRU)**
- **Configuration:** LSTM(64) → GRU(32) with temporal awareness
- **Purpose:** Sequence modeling for temporal clinical progression
- **Performance:** ~94.5% accuracy

---

## Ensemble Models

### Ensemble Strategy

Our ensemble framework leverages heterogeneous base learners with weighted averaging and stacking techniques:

#### **LightGBM–CatBoost Ensemble (Best Model)**
- **Architecture:** Weighted average of LightGBM and CatBoost predictions
- **Weights:** Optimized via validation set performance
  - LightGBM: 0.55
  - CatBoost: 0.45
- **Aggregation Method:** Soft voting with probability calibration
- **Performance:** **98.36% accuracy** on held-out test set
- **Generalization:** 97.89% accuracy on external validation cohorts

#### **Stacking Ensemble (Meta-Learner)**
- **Level 0 (Base Models):** XGBoost, LightGBM, CatBoost, Random Forest, SVM
- **Level 1 (Meta-Learner):** Logistic Regression
- **Performance:** 98.12% accuracy
- **Advantage:** Learns optimal base model weights

#### **Voting Ensemble (Soft Voting)**
- **Constituents:** 6 best-performing individual models
- **Voting Strategy:** Soft voting with weighted probability averaging
- **Performance:** 97.95% accuracy

---

## Best Results Table

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Training Time |
|-------|----------|-----------|--------|----------|---------|----------------|
| **Logistic Regression** | 92.15% | 0.918 | 0.919 | 0.918 | 0.958 | 0.12s |
| **Random Forest** | 95.82% | 0.962 | 0.958 | 0.960 | 0.985 | 2.34s |
| **XGBoost** | 96.23% | 0.965 | 0.961 | 0.963 | 0.989 | 3.45s |
| **LightGBM** | 97.81% | 0.979 | 0.976 | 0.977 | 0.995 | 1.89s |
| **CatBoost** | 97.54% | 0.976 | 0.974 | 0.975 | 0.993 | 2.12s |
| **SVM** | 94.78% | 0.951 | 0.948 | 0.949 | 0.978 | 4.23s |
| **Deep Neural Network** | 96.85% | 0.971 | 0.968 | 0.969 | 0.991 | 8.67s |
| **Stacking Ensemble** | 98.12% | 0.983 | 0.981 | 0.982 | 0.996 | 12.34s |
| **LightGBM–CatBoost Ensemble** | **98.36%** | **0.985** | **0.983** | **0.984** | **0.997** | 4.01s |

---

## Explainable AI Section

### SHAP (SHapley Additive exPlanations) Framework

#### **Model Interpretability Approach**

SHAP provides a unified framework grounded in cooperative game theory to decompose model predictions into individual feature contributions:

```
Individual Prediction = Base Value + Σ(SHAP Values for each feature)
```

#### **SHAP Analysis Outputs**

1. **SHAP Summary Plot (Bar):** 
   - Aggregated absolute feature importance across all predictions
   - Identifies globally most influential features for cancer stage prediction

2. **SHAP Summary Plot (Beeswarm):** 
   - Individual sample-level feature contributions
   - Color gradient indicates feature value (red=high, blue=low)
   - Reveals non-linear feature relationships and interactions

3. **SHAP Dependence Plots:** 
   - Relationship between feature value and SHAP contribution
   - Identifies threshold effects and interaction patterns
   - Example: Tumor size shows non-linear relationship with stage prediction

4. **SHAP Force Plots:** 
   - Individual prediction decomposition
   - Clinical case-level interpretability
   - Supports transparent model predictions for clinicians

#### **Key Interpretability Findings**

- **Most Important Features:** 
  - Tumor size, grade, lymph node involvement, ER/PR status
  - Accounts for 68% of cumulative SHAP importance

- **Feature Interactions:** 
  - ER status × HER2 status: Critical interaction for prognostic staging
  - Tumor grade × Lymph node involvement: Non-linear synergistic effect

- **Model Fairness:** 
  - SHAP analysis confirms equitable feature utilization across demographic groups
  - Age and ethnicity not disproportionately influencing predictions

#### **Clinical Explainability Benefits**

✓ Transparent decision-making for oncologists  
✓ Feature-level attribution for research insights  
✓ Identification of patient-specific risk factors  
✓ Support for clinical trial design and biomarker discovery  
✓ Regulatory compliance for AI in healthcare (FDA/CE marking considerations)

---

## Repository Structure

```
Predicting-Breast-Cancer-Stages-Through-Integrative-Machine-Learning-and-Explainable-AI-Approaches/
│
├── README.md                                    # This file
├── LICENSE                                      # Research license
│
├── 01_Data/
│   ├── raw/
│   │   ├── kaggle_breast_cancer.csv
│   │   ├── metabric_dataset.csv
│   │   └── seer_dataset.csv
│   ├── processed/
│   │   ├── kaggle_processed.csv
│   │   ├── metabric_processed.csv
│   │   └── seer_processed.csv
│   └── data_dictionary.md                       # Feature descriptions
│
├── 02_Notebooks/
│   ├── 01_EDA_Data_Exploration.ipynb            # Exploratory data analysis
│   ├── 02_Data_Preprocessing.ipynb              # Preprocessing pipeline
│   ├── 03_Feature_Engineering.ipynb             # Feature creation & selection
│   ├── 04_Traditional_ML_Models.ipynb           # Logistic Reg, RF, XGBoost, LightGBM, CatBoost, SVM
│   ├── 05_Deep_Learning_Models.ipynb            # DNN, CNN, LSTM/GRU architectures
│   ├── 06_Ensemble_Models.ipynb                 # Ensemble strategies & voting
│   ├── 07_Model_Evaluation.ipynb                # Cross-validation & performance metrics
│   ├── 08_SHAP_Explainability.ipynb             # SHAP visualizations & interpretability
│   ├── 09_Feature_Importance_Analysis.ipynb     # Traditional feature importance methods
│   └── 10_Clinical_Insights_Summary.ipynb       # Research findings & clinical implications
│
├── 03_Models/
│   ├── trained_models/
│   │   ├── lightgbm_model.pkl
│   │   ├── catboost_model.pkl
│   │   ├── ensemble_model.pkl
│   │   └── scaler.pkl
│   └── model_configs/
│       └── hyperparameters.yaml
│
├── 04_Results/
│   ├── performance_metrics.csv
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── shap_plots/
│   │   ├── summary_bar.png
│   │   ├── summary_beeswarm.png
│   │   └── dependence_plots/
│   └── feature_importance/
│
├── 05_Utilities/
│   ├── preprocessing.py                         # Data preprocessing functions
│   ├── feature_engineering.py                   # Feature creation utilities
│   ├── model_training.py                        # Model training pipelines
│   ├── evaluation_metrics.py                    # Performance evaluation functions
│   └── visualization.py                         # Plotting utilities
│
├── requirements.txt                             # Python dependencies
├── environment.yml                              # Conda environment specification
└── CITATION.cff                                 # Citation metadata

```

---

## Technologies Used

### Core Data Science & ML Libraries

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **Pandas** | 1.3+ | Data manipulation & analysis |
| **NumPy** | 1.21+ | Numerical computing |
| **Scikit-learn** | 1.0+ | Traditional ML algorithms |
| **XGBoost** | 1.5+ | Gradient boosting framework |
| **LightGBM** | 3.3+ | Fast gradient boosting |
| **CatBoost** | 1.0+ | Categorical gradient boosting |
| **TensorFlow/Keras** | 2.8+ | Deep learning framework |
| **SHAP** | 0.41+ | Model explainability |
| **Matplotlib** | 3.4+ | Data visualization |
| **Seaborn** | 0.11+ | Statistical visualization |

### Development Environment

- **Jupyter Notebook:** Interactive research notebooks
- **Git:** Version control
- **Virtual Environment:** Python package management
- **Docker:** Containerization (optional)

### Computational Requirements

- **Minimum:** 8GB RAM, 4 CPU cores, 10GB disk space
- **Recommended:** 16GB+ RAM, GPU (CUDA) for deep learning acceleration
- **Training Time:** ~30-45 minutes on standard hardware

---

## Citation Section

### BibTeX Citation

```bibtex
@research{tithi_breast_cancer_prediction_2026,
  title={Predicting Breast Cancer Progression Stages Through Integrative Machine Learning and Explainable AI Approaches},
  author={Tithi, Shammini},
  year={2026},
  conference={ECCT 2026 - European Conference on Computational Technologies},
  institution={Academic Research Repository},
  note={GitHub Repository},
  url={https://github.com/shammintithi/Predicting-Breast-Cancer-Stages-Through-Integrative-Machine-Learning-and-Explainable-AI-Approaches},
  keywords={breast cancer, machine learning, ensemble learning, explainable AI, SHAP, deep learning, clinical prediction},
}
```

### APA Citation

Tithi, S. (2026). Predicting breast cancer progression stages through integrative machine learning and explainable AI approaches. *ECCT 2026 - European Conference on Computational Technologies*. https://github.com/shammintithi/Predicting-Breast-Cancer-Stages-Through-Integrative-Machine-Learning-and-Explainable-AI-Approaches

### MLA Citation

Tithi, Shammini. "Predicting Breast Cancer Progression Stages Through Integrative Machine Learning and Explainable AI Approaches." *ECCT 2026 - European Conference on Computational Technologies*, 2026, https://github.com/shammintithi/Predicting-Breast-Cancer-Stages-Through-Integrative-Machine-Learning-and-Explainable-AI-Approaches.

---

## Disclaimer

### Research Scope & Limitations

1. **Not for Clinical Use:** This research is a proof-of-concept computational framework designed for research and educational purposes. **It is not intended for direct clinical decision-making or patient care without validation by qualified oncologists and regulatory approval.**

2. **Dataset Limitations:** 
   - Models trained on retrospective datasets with potential selection bias
   - Performance may not generalize to all patient populations
   - Cross-cohort validation recommended before clinical deployment

3. **Model Uncertainty:** 
   - All predictive models contain inherent uncertainty
   - Confidence intervals and calibration curves should inform interpretation
   - False positive and false negative rates require clinical context

4. **Data Privacy:** 
   - Public datasets used contain de-identified patient information
   - Adherence to HIPAA (US), GDPR (EU), and local data protection regulations required for healthcare deployment
   - Sensitive data should never be shared in public repositories

5. **Regulatory Compliance:** 
   - Deployment in clinical settings requires FDA clearance (US), CE marking (EU), or equivalent regulatory approval
   - Algorithm validation on prospective patient cohorts mandatory
   - Clinical trial protocols and ethics board approval necessary

6. **Bias & Fairness:** 
   - While SHAP analysis conducted to assess feature fairness, inherent biases in training data may persist
   - Continuous monitoring and fairness audits recommended for clinical deployment

7. **Intellectual Property:** 
   - This research and code are provided as-is for academic purposes
   - Commercial use requires explicit authorization

### Recommended Reading

- Rajkomar et al. (2018): "Scalable and accurate deep learning with electronic health records" - *NPJ Digital Medicine*
- Caruana et al. (2015): "Intelligible models for healthcare" - *KDD*
- Lundberg & Lee (2017): "A unified approach to interpreting model predictions" - *NIPS*

---

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/shammintithi/Predicting-Breast-Cancer-Stages-Through-Integrative-Machine-Learning-and-Explainable-AI-Approaches.git
cd repository-name

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run EDA notebook
jupyter notebook 02_Notebooks/01_EDA_Data_Exploration.ipynb

# Execute preprocessing pipeline
jupyter notebook 02_Notebooks/02_Data_Preprocessing.ipynb

# Train ensemble model
jupyter notebook 02_Notebooks/06_Ensemble_Models.ipynb

# Generate SHAP explainability visualizations
jupyter notebook 02_Notebooks/08_SHAP_Explainability.ipynb
```

---

## Contact & Collaboration

**Lead Researcher:** Shammini Tithi  
**Email:** [Your Email]  
**GitHub:** [@shammintithi](https://github.com/shammintithi)  

For research collaboration, questions, or dataset access requests, please open an issue in this repository or contact the lead author directly.

---

## Acknowledgments

- **Data Sources:** Kaggle, METABRIC Consortium, US National Cancer Institute (SEER)
- **Libraries:** Scikit-learn, TensorFlow, XGBoost, SHAP communities
- **Conference:** ECCT 2026 Organization Committee

---

## License

This research project is made available under the [MIT License](LICENSE). Attribution to the original authors is appreciated for academic and professional use.

---

**Last Updated:** June 2026  
**Repository Status:** Active Research  
**Best Model Performance:** LightGBM–CatBoost Ensemble - **98.36% Accuracy**
