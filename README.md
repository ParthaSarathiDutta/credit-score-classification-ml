
# Credit Score Classification Using ML

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-yellow)

This repository presents a machine learning-based pipeline for classifying credit scores into categorical buckets such as *Poor*, *Standard*, and *Good*. It compares classical ML models including Logistic Regression, SVM, Random Forest, XGBoost, and MLP on a structured tabular dataset.

---

## Overview

Credit scoring plays a critical role in determining an individual’s creditworthiness. This project develops and benchmarks multiple supervised ML models to classify credit score categories based on demographic and financial indicators.

**Key Features Used**:
- Payment history
- Credit mix
- Credit utilization
- Delinquencies
- Annual income

**Target**: Categorical credit score classification (Poor, Standard, Good)

---

## Models

- **Logistic Regression**: Linear classifier for baseline performance
- **SVM**: Margin-based classifier for high-dimensional separation
- **Random Forest**: Ensemble decision trees for robust feature importance
- **XGBoost**: Gradient-boosted model for superior accuracy
- **MLP**: Multi-layer perceptron trained on scaled features

---

## Dataset

- **Source**: [Kaggle / Proprietary]
- **File**: `credit_score_data.csv`
- **Shape**: ~10,000 samples × 20+ features
- **Preprocessing**:
  - One-hot encoding for categorical features
  - Missing value imputation
  - Feature normalization using MinMaxScaler
  - Train/test split (80:20)

---

## Results

| Model              | Accuracy | F1 Score |
|-------------------|----------|----------|
| Logistic Regression | 85%     | 0.84     |
| SVM                 | 87%     | 0.86     |
| Random Forest       | 91%     | 0.91     |
| XGBoost             | 92%     | 0.92     |
| **MLP Classifier**  | **94%** | **0.93** |

> MLP performed best across all evaluation metrics.

---

## Repository Structure

```

credit-score-classification-ml/
├── model_training.ipynb             # Jupyter notebook with data preprocessing, training, and evaluation
├── model_training.py                # Python script version for reproducibility
├── CreditScoreClassification.pdf    # Final project report
├── CreditScoreClassification.pptx   # Project presentation slides
├── requirements.txt                 # Python dependencies (Scikit-learn, XGBoost, Pandas, etc.)
├── LICENSE                          # MIT License
└── README.md                        # Project overview, usage guide, and results

````

---

## Running Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/credit-score-classification-ml.git
   cd credit-score-classification-ml
````

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook model_training.ipynb
   ```

---

## Dependencies

* Python 3.8+
* NumPy, Pandas
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn

---

## Contact

**Partha Sarathi Dutta**
AI Researcher, Argonne National Laboratory
[pdutta3@uic.edu](mailto:pdutta3@uic.edu)
[GitHub](https://github.com/ParthaSarathiDutta)

---

## Acknowledgments

This work was carried out in collaboration with **Stoyan Stoyanov**, **Varun Iyer** and **Partha Sarathi Dutta** under the guidance of **Prof. Xinhua Zhang** at the University of Illinois Chicago.

---

## License

Distributed under the MIT License.


