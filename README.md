# Diabetes-Prediction
# Harnessing XGBoost and Other Techniques for Enhanced Diabetes Diagnosis

[cite_start]This repository contains the research paper "Harnessing XGBoost and Other Techniques for enhanced diabetes diagnosis"[cite: 1]. The study investigates and compares various machine learning models for accurately predicting diabetes risk.

---

## Abstract

The paper addresses the rising incidence of diabetes and the need for accurate, reliable risk prediction techniques. It explores several machine learning (ML) models—including logistic regression, decision trees, random forest, and support vector machines (SVM)—to analyze patient data and identify significant risk factors. [cite_start]The goal is to develop an easy-to-interpret model for early detection, enabling timely intervention and improved patient outcomes[cite: 1].

---

## Research Methodology

### Dataset

The analysis was conducted using the **PIMA Indians Diabetes Dataset**. This dataset is widely used in medical research and includes health information from 768 female Pima Indian patients aged 21 and older. [cite_start]It contains eight input features and one binary outcome (diabetic or non-diabetic)[cite: 1].

The features are:
* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI (Body Mass Index)
* Diabetes Pedigree Function
* Age

### Model Pipeline

[cite_start]The study followed a standard machine learning pipeline[cite: 1]:
1.  **Data Preprocessing:** Cleaning the raw data by handling null/duplicate values, smoothing noise, and normalization.
2.  **Train-Test Split:** Splitting the enriched data into training and testing sets.
3.  **Applying ML Models:** Training various classifiers on the data.
4.  **Hyperparameter Tuning:** Optimizing model parameters using techniques like GridSearch.
5.  **Model Evaluation:** Assessing model performance using metrics like Accuracy, Precision, Recall, and F1-Score.

---

## Models & Key Results

The paper analyzed several models, with a focus on ensemble techniques.

### Models Explored
* [cite_start]**K-Nearest Neighbors (KNN):** A simple, distance-based classification algorithm[cite: 1].
* [cite_start]**Decision Tree:** An interpretable, tree-based model that splits data based on feature thresholds[cite: 1].
* **XGBoost (Extreme Gradient Boosting):** A powerful ensemble method that builds sequential decision trees, correcting the residuals of preceding ones. [cite_start]It includes robust regularization to prevent overfitting[cite: 1].
* [cite_start]**CatBoost (Categorical Boost):** Another advanced gradient boosting algorithm well-suited for both numerical and categorical data[cite: 1].

### Performance Comparison

| Model | Accuracy (%) | Limitations |
| :--- | :---: | :--- |
| KNN | [cite_start]80% [cite: 1] | [cite_start]Performance drops with high-dimensional data[cite: 1]. |
| Decision Tree | [cite_start]79.87% [cite: 1] | [cite_start]Prone to overfitting without pruning[cite: 1]. |
| **XGBoost** | [cite_start]**97%** [cite: 1] | [cite_start]Uses L1 and L2 regularization to reduce overfitting[cite: 1]. |
| **CatBoost** | [cite_start]**96%** [cite: 1] | [cite_start]Uses L1 and L2 regularization to reduce overfitting[cite: 1]. |

### ROC-AUC Scores

The models' ability to discriminate between diabetic and non-diabetic patients was evaluated using the Receiver Operating Characteristic (ROC) curve.

* [cite_start]**XGBoost:** Achieved an **AUC (Area Under the Curve) of 0.94**, indicating excellent discriminative capability[cite: 1].
* [cite_start]**CatBoost:** Achieved an **AUC of 0.93**, also indicating a high degree of classification performance[cite: 1].

---

## Conclusion

The comparative study demonstrates that **XGBoost systematically surpasses standard models** like KNN and Decision Trees. This superior performance is attributed to its advanced regularization techniques and its ability to capture complex nonlinear dependencies in the data. [cite_start]The paper concludes that XGBoost, combined with effective preprocessing, provides a dramatically improved tool for early diabetes detection and management[cite: 1].
