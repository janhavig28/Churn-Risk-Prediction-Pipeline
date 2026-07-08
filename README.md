# Churn Risk Prediction

A machine learning project to predict whether a telecom customer is likely to churn (leave the service), based on their account and usage details.

## Dataset

Telco Customer Churn dataset (7,043 customer records) with features like tenure, monthly charges, contract type, and the services they've subscribed to.

## What I did

- Cleaned the data (handled missing values in `TotalCharges`, dropped the customer ID column)
- Did EDA — checked distributions, outliers, and correlations for numerical features, and count plots for categorical ones
- Encoded categorical columns using Label Encoding
- The target column (`Churn`) was imbalanced, so I used SMOTE to balance the training data
- Trained and compared three models: Decision Tree, Random Forest, and XGBoost
- Tuned both Random Forest and XGBoost using RandomizedSearchCV — their tuned performance came out nearly identical, so I went with Random Forest
- Adjusted the classification threshold (0.35 instead of the default 0.5) to improve recall, since catching customers who are about to churn matters more than overall accuracy
- Used SHAP to understand which features were driving the model's predictions
- Saved the trained model and encoders as pickle files, and tested it on a sample input

## Tech stack

- Python, pandas, numpy
- scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- SHAP for explainability
- matplotlib, seaborn for EDA

## Results

Random Forest gave the best cross-validation accuracy among the three models. After threshold tuning, recall on the churn class improved noticeably compared to the default 0.5 threshold.

## Status

Core model training, tuning, and explainability are done. Right now it's a notebook — no UI or deployment yet.
