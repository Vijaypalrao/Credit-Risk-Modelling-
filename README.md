# Credit Risk Prediction System

**Tags:**  
credit-risk, machine-learning, xgboost, logistic-regression, probability-of-default, expected-loss, streamlit, banking, finance, data-science

---

# Project Overview

This project develops an end-to-end Credit Risk Prediction System to estimate the **Probability of Default (PD)** for loan applicants using historical consumer loan data.

Two machine learning models, **Logistic Regression** and **XGBoost**, are developed and compared to identify the better-performing model. The final model is used to predict default probability, support loan approval decisions, estimate Expected Loss (EL), and is deployed as an interactive Streamlit web application.

---

# Problem Statement

Financial institutions must balance loan approvals with credit risk.

The objective of this project is to:

- Predict the Probability of Default (PD) of loan applicants.
- Compare multiple machine learning models.
- Approve or reject loans using a PD threshold.
- Estimate Expected Loss (EL) for risk assessment.
- Deploy the final model as a web application.

---

# Project Phases

## Phase 1 — Data Preparation & Exploratory Data Analysis (EDA)

- Data cleaning and preprocessing
- Missing value treatment
- Outlier detection
- Feature analysis
- Default rate analysis across borrower segments
- Data visualization

---

## Phase 2 — Model Development

### Logistic Regression

- Data preprocessing pipeline
- Median and mode imputation
- One-Hot Encoding
- Model training and evaluation

### XGBoost

- SMOTE for class imbalance
- Model training
- Hyperparameter tuning
- Performance comparison with Logistic Regression

---

## Phase 3 — Model Evaluation & Business Decision

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Brier Score
- R² Score

The best-performing model is selected to:

- Predict Probability of Default (PD)
- Estimate Expected Loss (EL = PD × LGD × EAD)
- Make loan approval decisions using a **10% PD cutoff**

Business Rule:

- **PD ≤ 10% → APPROVE LOAN**
- **PD > 10% → REJECT LOAN**

---

## Phase 4 — Web Application Deployment

The final XGBoost model is saved using Joblib and deployed using **Streamlit**.

The web application allows users to:

- Enter applicant details
- Predict Probability of Default
- View Loan Approval / Rejection Decision
- Calculate Expected Loss instantly

The application is deployed online and accessible through a public Streamlit URL.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit
- Matplotlib

---

# Project Structure

```text
Credit_Risk_Web_App/
│
├── app.py
├── best_credit_risk_model.pkl
├── requirements.txt
├── README.md
```

---

# Outputs

The deployed application provides:

- Probability of Default (PD)
- Loan Approval / Rejection Decision
- Expected Loss (EL)

---

# Future Improvements

- Risk score categorization (Low, Medium, High)
- Batch prediction using CSV upload
- Loan portfolio dashboard
- Model monitoring and periodic retraining
