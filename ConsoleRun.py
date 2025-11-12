import os
import pickle
import numpy as np
import pandas as pd
# import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# with open("./models/income_model.pkl", "rb") as f:
#     income = pickle.load(f)
# with open("./models/city_tier_model.pkl", "rb") as f:
#     city_tier = pickle.load(f)
# with open("./models/financial_behavior_model.pkl", "rb") as f:
#     income_resources = pickle.load(f)
# with open("./models/profession_model.pkl", "rb") as f:
#     profession = pickle.load(f)
# with open("./models/dependents_model.pkl", "rb") as f:
#     dependents = pickle.load(f)

index=['Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries',
                'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
                'Education', 'Miscellaneous', 'Desired_Savings', 'Occupation_Retired',
                'Occupation_Self_Employed', 'Occupation_Student', 'City_Tier_Tier_2',
                'City_Tier_Tier_3']

def income_predictor(new_data):
        

        new_data_encoded = pd.get_dummies(new_data, columns=['Occupation', 'City_Tier'], drop_first=True)
        new_data_encoded = new_data_encoded.reindex(columns=index, fill_value=0)
        prediction = income.predict(new_data_encoded)
        print("Predicted Income:", prediction[0])



new_data=pd.DataFrame({
    'Age': [26],
    'Occupation': ['Student'],
    'City_Tier': ['Tier_1'],
    'Dependents': [2],
    'Rent': [15000],
    'Loan_Repayment': [5000],
    'Insurance': [2000],
    'Groceries': [8000],
    'Transport': [3000],
    'Eating_Out': [2000],
    'Entertainment': [1500],
    'Utilities': [2500],
    'Healthcare': [1000],
    'Education': [2000],
    'Miscellaneous': [1200],
    'Desired_Savings': [100]
})



def predict_city_tier(new_data):
    
    city_model = pickle.load(open("./models/city_tier_model.pkl", "rb"))
    preprocessor = pickle.load(open("./models/city_tier_preprocessor.pkl", "rb"))
    label_encoder = pickle.load(open("./models/city_tier_labelencoder.pkl", "rb"))

    if isinstance(new_data, dict):
        df = pd.DataFrame([new_data])
    elif isinstance(new_data, pd.DataFrame):
        df = new_data.copy()
    else:
        raise ValueError("Input must be a dict or pandas DataFrame")

    df['edu_ratio'] = df['Education'] / (df['Income'] + 1)
    df['health_ratio'] = df['Healthcare'] / (df['Income'] + 1)
    df['grocery_ratio'] = df['Groceries'] / (df['Income'] + 1)
    df['entertainment_ratio'] = df['Entertainment'] / (df['Income'] + 1)
    df['total_spending_ratio'] = (df['Groceries'] + df['Education'] + df['Healthcare'] + df['Utilities']) / (df['Income'] + 1)
    df['loan_ratio'] = df['Loan_Repayment'] / (df['Income'] + 1)
    df['savings_ratio'] = df['Disposable_Income'] / (df['Income'] + 1)
    df['expense_ratio'] = (df['Groceries'] + df['Education'] + df['Healthcare']) / (df['Income'] + 1)

    selected_cols = [
        'Income', 'Age', 'Occupation',
        'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Education',
        'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous',
        'edu_ratio', 'health_ratio', 'grocery_ratio',
        'entertainment_ratio', 'total_spending_ratio',
        'loan_ratio', 'savings_ratio', 'expense_ratio'
    ]
    X_new = df[selected_cols]
    X_new_encoded = preprocessor.transform(X_new)
    y_pred = city_model.predict(X_new_encoded)
    predicted_label = label_encoder.inverse_transform(y_pred)
    return predicted_label[0]

new_person = {
    'Income': 60000,
    'Age': 32,
    'Occupation': 'Engineer',
    'Rent': 10000,
    'Loan_Repayment': 1500,
    'Insurance': 800,
    'Groceries': 3500,
    'Transport': 1000,
    'Education': 600,
    'Healthcare': 900,
    'Entertainment': 1200,
    'Utilities': 1800,
    'Miscellaneous': 700,
    'Disposable_Income': 12000  
}


def load_dependents_model(model_path="./models/dependents_model.pkl", preprocessor_path="./models/dependents_preprocessor.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

def predict_dependents(new_data_df, model, preprocessor):
    for col in ['Occupation', 'City_Tier']:
        new_data_df[col] = new_data_df[col].astype(str)
    X_encoded = preprocessor.transform(new_data_df)
    y_pred = model.predict(X_encoded)
    return y_pred + 1

model, preprocessor = load_dependents_model()

new_data = pd.DataFrame([{
    "Income": 60000,
    "Age": 35,
    "Occupation": "Student",
    "City_Tier": 'Tier_2',
    "Rent": 5000,
    "Loan_Repayment": 5000,
    "Insurance": 2000,
    "Groceries": 8000,
    "Transport": 3000,
    "Education": 4000,
    "Healthcare": 2500,
    "Entertainment": 2000,
    "Utilities": 3000,
    "Miscellaneous": 1000
}])

predicted_dependents = predict_dependents(new_data, model, preprocessor)
print('Dependents: ',predicted_dependents[0])




# -------------------------------
# 1️⃣ Load model, preprocessor, label encoder
# -------------------------------
with open("./models/dependents_improved_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("./models/dependents_improved_preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("./models/dependents_improved_labelencoder.pkl", "rb") as f:
    le = pickle.load(f)

# -------------------------------
# 2️⃣ Define prediction function
# -------------------------------
def predict_improved_dependents(new_data: dict):
    df_new = pd.DataFrame([new_data])

    # Feature engineering
    df_new['edu_ratio'] = df_new['Education'] / (df_new['Income'] + 1)
    df_new['health_ratio'] = df_new['Healthcare'] / (df_new['Income'] + 1)
    df_new['grocery_ratio'] = df_new['Groceries'] / (df_new['Income'] + 1)
    df_new['entertainment_ratio'] = df_new['Entertainment'] / (df_new['Income'] + 1)
    df_new['total_spending_ratio'] = (
        df_new['Groceries'] + df_new['Education'] + df_new['Healthcare'] + df_new['Utilities']
    ) / (df_new['Income'] + 1)
    df_new['loan_ratio'] = df_new['Loan_Repayment'] / (df_new['Income'] + 1)
    df_new['savings_ratio'] = df_new['Disposable_Income'] / (df_new['Income'] + 1)
    df_new['expense_ratio'] = (
        df_new['Groceries'] + df_new['Education'] + df_new['Healthcare']
    ) / (df_new['Income'] + 1)

    # Columns in same order as training
    selected_cols = [
        'Income', 'Age', 'Occupation', 'City_Tier',
        'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Education',
        'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous',
        'edu_ratio', 'health_ratio', 'grocery_ratio',
        'entertainment_ratio', 'total_spending_ratio',
        'loan_ratio', 'savings_ratio', 'expense_ratio'
    ]

    X_new = df_new[selected_cols]

    # Transform with preprocessor
    X_new_encoded = preprocessor.transform(X_new)

    # Predict
    y_pred_encoded = model.predict(X_new_encoded)

    # Decode label
    y_pred = le.inverse_transform(y_pred_encoded)

    return y_pred[0]

# -------------------------------
# 3️⃣ Example usage
# -------------------------------

new_person = {
    "Income": 60000,
    "Age": 35,
    "Occupation": "Student",
    "City_Tier": 'Tier_2',
    "Rent": 5000,
    "Loan_Repayment": 5000,
    "Insurance": 2000,
    "Groceries": 8000,
    "Transport": 3000,
    "Education": 4000,
    "Healthcare": 2500,
    "Entertainment": 2000,
    "Utilities": 3000,
    "Miscellaneous": 1000,
    "Disposable_Income": 12000
}

predicted_dependents = predict_improved_dependents(new_person)
print("Predicted Dependents:", predicted_dependents)



def predict_city_tier_person(data: dict):
    with open("./models/city_tier_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("./models/city_tier_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("./models/city_tier_labelencoder.pkl", "rb") as f:
        le = pickle.load(f)

    df = pd.DataFrame([data])

    df['edu_ratio'] = df['Education'] / (df['Income'] + 1)
    df['health_ratio'] = df['Healthcare'] / (df['Income'] + 1)
    df['grocery_ratio'] = df['Groceries'] / (df['Income'] + 1)
    df['entertainment_ratio'] = df['Entertainment'] / (df['Income'] + 1)
    df['total_spending_ratio'] = (
        df['Groceries'] + df['Education'] + df['Healthcare'] + df['Utilities']
    ) / (df['Income'] + 1)
    df['loan_ratio'] = df['Loan_Repayment'] / (df['Income'] + 1)
    df['savings_ratio'] = df['Disposable_Income'] / (df['Income'] + 1)
    df['expense_ratio'] = (
        df['Groceries'] + df['Education'] + df['Healthcare']
    ) / (df['Income'] + 1)

    selected_cols = [
        'Income', 'Age', 'Occupation',
        'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Education',
        'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous',
        'edu_ratio', 'health_ratio', 'grocery_ratio',
        'entertainment_ratio', 'total_spending_ratio',
        'loan_ratio', 'savings_ratio', 'expense_ratio'
    ]

    X = df[selected_cols]
    X_encoded = preprocessor.transform(X)
    y_pred = model.predict(X_encoded)
    city_tier = le.inverse_transform(y_pred)
    return city_tier

# Example usage:
new_person = {
    'Income': 50000,
    'Age': 35,
    'Occupation': 'Salaried',
    'Rent': 12000,
    'Loan_Repayment': 5000,
    'Insurance': 1500,
    'Groceries': 8000,
    'Transport': 3000,
    'Education': 2000,
    'Healthcare': 1000,
    'Entertainment': 500,
    'Utilities': 2000,
    'Miscellaneous': 1000,
    'Disposable_Income': 15000
}

predicted_tier = predict_city_tier_person(new_person)
print("Predicted City Tier:", predicted_tier[0])






# Function to load model and predict for new data
def predict_income(new_data: dict):
    with open("./models/income_model.pkl", "rb") as f:
        model = pickle.load(f)

    df_new = pd.DataFrame([new_data])

    df_new_encoded = pd.get_dummies(df_new, columns=['Occupation', 'City_Tier'], drop_first=True)

    df_new_encoded = df_new_encoded.reindex(columns=index, fill_value=0)

    prediction = model.predict(df_new_encoded)
    return prediction[0]

# Example usage
new_person = {
    'Age': 26,
    'Occupation': 'Student',
    'City_Tier': 'Tier_1',
    'Dependents': 2,
    'Rent': 15000,
    'Loan_Repayment': 5000,
    'Insurance': 2000,
    'Groceries': 8000,
    'Transport': 3000,
    'Eating_Out': 2000,
    'Entertainment': 1500,
    'Utilities': 2500,
    'Healthcare': 1000,
    'Education': 2000,
    'Miscellaneous': 1200,
    'Desired_Savings': 100
}

predicted_income = predict_income(new_person)
print("Predicted Income:", predicted_income)


with open("./models/financial_behavior_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("./models/financial_preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

target_cols = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education',
    'Miscellaneous', 'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income'
]

def predict_financial_behavior(input_dict):
    X_new = pd.DataFrame([input_dict])
    X_proc = preprocessor.transform(X_new)
    preds = model.predict(X_proc)[0]
    return dict(zip(target_cols, preds))

sample_input = {
    "Income": 85000,
    "City_Tier": "Tier 2",
    "Occupation": "Private Job",
    "Dependents": 2
}

predictions = predict_financial_behavior(sample_input)
for k, v in predictions.items():
    print(f"{k}: {v:.2f}")



def predict_profession(new_data):
    """
    Predict the profession for new input data.
    
    Parameters:
        new_data (dict): Dictionary with keys matching the feature columns:
            'Income', 'Age', 'City_Tier', 'Rent', 'Loan_Repayment', 'Insurance',
            'Groceries', 'Transport', 'Education', 'Healthcare', 'Entertainment',
            'Utilities', 'Miscellaneous', 'Dependents'
    
    Returns:
        str: Predicted profession
    """
    model = pickle.load(open("./models/profession_model.pkl", "rb"))
    preprocessor = pickle.load(open("./models/profession_preprocessor.pkl", "rb"))
    le = pickle.load(open("./models/profession_labelencoder.pkl", "rb"))
    
    X_new = pd.DataFrame([new_data])
    X_encoded = preprocessor.transform(X_new)
    pred_class = model.predict(X_encoded)[0]
    pred_label = le.inverse_transform([pred_class])[0]
    
    return pred_label

# Example usage
sample_input = {
    'Income': 85000,
    'Age': 35,
    'City_Tier': 'Tier 2',
    'Rent': 12000,
    'Loan_Repayment': 5000,
    'Insurance': 3000,
    'Groceries': 8000,
    'Transport': 2000,
    'Education': 4000,
    'Healthcare': 2500,
    'Entertainment': 1500,
    'Utilities': 3000,
    'Miscellaneous': 1000,
    'Dependents': 2
}

predicted_profession = predict_profession(sample_input)
print("Predicted Profession:", predicted_profession)
