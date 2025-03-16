# Bank Marketing Campaign Prediction

## Project Overview

This project aims to predict the success of bank marketing campaigns, specifically the likelihood that a customer will subscribe to a term deposit after being contacted by a bank. We use machine learning models, such as Random Forest and XGBoost, to build a classification model for predicting the outcome of marketing campaigns based on customer and social/economic data.

The dataset used in this project is the **Bank Marketing** dataset, originally described in the paper by Moro et al. (2014). The goal is to predict whether a client will subscribe to a term deposit (binary classification: yes/no) based on various features such as customer demographics, contact communication type, and social/economic context attributes.

## Dataset

### Source

This dataset is publicly available for research and was introduced in the paper:

**Moro et al. (2014)**: *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, In press. [DOI: 10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001)

### Dataset Details
- **Instances**: 41,188
- **Features**: 20 input features + 1 output feature (target)
- **Target**: `y` - whether the client subscribed to a term deposit (binary classification: "yes" or "no")
- **Input Features**: Includes customer attributes (age, job, marital status, education, loan status, etc.), campaign information (number of contacts, outcome of previous campaign), and social/economic context attributes (employment rate, consumer price index, etc.)
- **Data Files**:
  - `bank-additional-full.csv`: Full dataset
  - `bank-additional.csv`: A random 10% sample of the full dataset

### Data Information

- **Input Variables**:
  - **Age** (numeric)
  - **Job** (categorical): Type of job (e.g., admin, technician, student, etc.)
  - **Marital** (categorical): Marital status
  - **Education** (categorical): Education level
  - **Default** (categorical): Credit default status
  - **Housing** (categorical): Housing loan status
  - **Loan** (categorical): Personal loan status
  - **Contact** (categorical): Communication type (cellular or telephone)
  - **Month** (categorical): Last contact month
  - **Day of Week** (categorical): Last contact day of the week
  - **Duration** (numeric): Last contact duration in seconds (important but can lead to data leakage)
  - **Campaign** (numeric): Number of contacts during the current campaign
  - **Pdays** (numeric): Days since last contact in a previous campaign
  - **Previous** (numeric): Number of contacts before the current campaign
  - **Poutcome** (categorical): Outcome of the previous marketing campaign
  - **Social and Economic Context Attributes**:
    - Employment rate
    - Consumer price index
    - Consumer confidence index
    - Euribor 3-month rate
    - Number of employees

- **Output Variable**:
  - **y**: Whether the client subscribed to a term deposit (binary: "yes" or "no")

### Missing Data

The dataset contains several missing values in categorical attributes, coded as the `"unknown"` label. These missing values can be handled using imputation or treated as a separate class.

## Model Overview

The goal of this project is to build a classification model to predict whether a client will subscribe to a term deposit based on the input features. The project involves:

- Data preprocessing (handling missing values, encoding categorical features, scaling numerical features)
- Model training using Random Forest and XGBoost
- Model evaluation using accuracy, ROC-AUC score, and confusion matrix
- Hyperparameter tuning for XGBoost using GridSearchCV

### Models Used:
- **Random Forest Classifier**
- **XGBoost Classifier**

### Performance Metrics:
- **Accuracy**
- **ROC-AUC Score**
- **Confusion Matrix**
- **Classification Report**

## Results

### Model Performance

- **Random Forest**:
  - Accuracy: `0.9516`
  - ROC-AUC Score: `0.6004`

- **XGBoost (before tuning)**:
  - Accuracy: `0.9526`
  - ROC-AUC Score: `0.6585`

- **XGBoost (after tuning)**:
  - Best Hyperparameters: `{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}`
  - Best ROC-AUC Score: `0.9632`
  - XGBoost Best Model Accuracy: `0.9557`
  - XGBoost Best Model ROC-AUC Score: `0.6524`

### Visualizations
The project also includes visualizations of model performance, such as ROC curves, to compare the performance of the models before and after hyperparameter tuning.

## Installation

### Requirements

1. Python 3.x
2. Install required packages using `pip`:

```bash
pip install -r requirements.txt
