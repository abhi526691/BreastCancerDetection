# Breast Cancer Wisconsin (Diagnostic) Dataset Analysis

## Overview

This repository contains the implementation and analysis of various machine learning models on the Breast Cancer Wisconsin (Diagnostic) Dataset. The aim is to classify tumors as either malignant or benign based on features computed from digitized images of fine needle aspirates of breast masses.

## Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Features**: 30 real-valued features computed for each cell nucleus
- **Target**: Binary classification (Malignant = 1, Benign = 0)
- **Instances**: 569

## Models Implemented

1. **Support Vector Machine (SVM)**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

Each model was tuned using GridSearchCV or RandomizedSearchCV with 5-fold cross-validation to optimize their performance.

## Evaluation Metrics

Models were evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC score.

## Analysis Summary

After thorough analysis and tuning, it was found that all three models performed well on the test data. However, considering the precision, recall, and overall F1-score, especially in a medical context where both false positives and false negatives carry significant consequences, SVM with Randomized Search CV emerged as the most balanced algorithm.

## Usage

The code for data preprocessing, model implementation, hyperparameter tuning, and evaluation is provided in Jupyter notebooks within this repository. Ensure to install the required libraries listed in `requirements.txt` before running the notebooks.

## Setup and Installation

```bash
pip install -r requirements.txt
```

## Conclusion

This study showcases the effectiveness of machine learning algorithms in classifying breast cancer tumors using diagnostic images. It underscores the importance of hyperparameter tuning in improving model accuracy and precision, providing valuable insights for medical diagnosis and treatment planning.
