
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

- **Source**: Proprietary financial dataset (not publicly available)
- **File**: `credit_score_data.csv`
- **Shape**: ~10,000 samples × 20+ features
- **Preprocessing**:
  - One-hot encoding for categorical features
  - Missing value imputation
  - Feature normalization using MinMaxScaler
  - Train/test split (80:20)

---

## Results

| Model               | Train Accuracy (%) | Test Accuracy (%) | Test F1 Score (%) |
|---------------------|--------------------|-------------------|-------------------|
| Random Forest       | 99.91              | 77.96             | 75.91             |
| XGBoost             | **100.00**         | **79.12**         | **78.29**         |
| Logistic Regression | 66.31              | 61.61             | 57.12             |
| SVM (Polynomial)    | 67.67              | 67.64             | 66.17             |
| SVM (RBF)           | 71.67              | 67.79             | 64.42             |
| MLP                 | 76.65              | 70.80             | 69.23             |

> **Note:** XGBoost achieved the best overall performance on the test set. Random Forest shows high training accuracy but lower generalization, indicating potential overfitting.


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

```

---

## Running Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/credit-score-classification-ml.git
   cd credit-score-classification-ml
   ```

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


