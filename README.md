# Churn Risk Prediction

A machine learning project to predict whether a telecom customer is likely to churn (leave the service), based on their account and usage details.

## Dataset

Telco Customer Churn dataset with 7,043 customer records and features such as tenure, monthly charges, contract type, and subscribed services.

## What I Did

- Cleaned the data by handling missing values in `TotalCharges` and dropping the customer ID column.
- Performed EDA to analyze distributions, outliers, correlations, and categorical feature frequencies.
- Encoded categorical columns using Label Encoding.
- Addressed class imbalance in the `Churn` target using SMOTE on the training data.
- Trained and compared three models: Decision Tree, Random Forest, and XGBoost.
- Tuned Random Forest and XGBoost using `RandomizedSearchCV`. Their tuned performance was nearly identical, so Random Forest was selected.
- Adjusted the classification threshold from the default `0.5` to `0.35` to improve recall, since identifying customers likely to churn was prioritized over overall accuracy.
- Used SHAP to understand which features contributed to the model's predictions.
- Saved the trained model and encoders as pickle files and tested the model on a sample input.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib
- Seaborn

## Results

Random Forest achieved the best cross-validation accuracy among the three models.

After threshold tuning, recall for the churn class improved compared with the default `0.5` threshold.

## Status

Core model training, hyperparameter tuning, threshold optimization, and explainability are complete.

The project is currently implemented as a Jupyter notebook. No UI or deployment has been added yet.