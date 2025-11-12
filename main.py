import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models and preprocessors globally to avoid reloading on every request
income_model = pickle.load(open("./models/income_model.pkl", "rb"))
city_tier_model = pickle.load(open("./models/city_tier_model.pkl", "rb"))
city_tier_preprocessor = pickle.load(open("./models/city_tier_preprocessor.pkl", "rb"))
city_tier_labelencoder = pickle.load(open("./models/city_tier_labelencoder.pkl", "rb"))

dependents_model = pickle.load(open("./models/dependents_model.pkl", "rb"))
dependents_preprocessor = pickle.load(open("./models/dependents_preprocessor.pkl", "rb"))

financial_model = pickle.load(open("./models/financial_behavior_model.pkl", "rb"))
financial_preprocessor = pickle.load(open("./models/financial_preprocessor.pkl", "rb"))

profession_model = pickle.load(open("./models/profession_model.pkl", "rb"))
profession_preprocessor = pickle.load(open("./models/profession_preprocessor.pkl", "rb"))
profession_labelencoder = pickle.load(open("./models/profession_labelencoder.pkl", "rb"))


model_path = "./models/dependents_improved_model.pkl"
preprocessor_path = "./models/dependents_improved_preprocessor.pkl"
le_path = "./models/dependents_improved_labelencoder.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

with open(le_path, "rb") as f:
    le = pickle.load(f)


expected_features = [
    'Income', 'Age', 'Occupation', 'City_Tier',
    'Rent', 'Loan_Repayment', 'Insurance',
    'Groceries', 'Transport', 'Education',
    'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous'
]

income_index = ['Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries',
                'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
                'Education', 'Miscellaneous', 'Desired_Savings', 'Occupation_Retired',
                'Occupation_Self_Employed', 'Occupation_Student', 'City_Tier_Tier_2',
                'City_Tier_Tier_3']

financial_target_cols = [
    'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education',
    'Miscellaneous', 'Desired_Savings_Percentage', 'Desired_Savings', 'Disposable_Income'
]


@app.route("/predict/income", methods=["POST"])
def predict_income():
    data = request.json
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df, columns=['Occupation', 'City_Tier'], drop_first=True)
    df_encoded = df_encoded.reindex(columns=income_index, fill_value=0)
    pred = income_model.predict(df_encoded)[0]
    return jsonify({"predicted_income": float(pred)})


@app.route("/predict/city_tier", methods=["POST"])
def predict_city_tier():
    data = request.json
    df = pd.DataFrame([data])
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

    X = df[selected_cols]
    X_encoded = city_tier_preprocessor.transform(X)
    y_pred = city_tier_model.predict(X_encoded)
    city_tier = city_tier_labelencoder.inverse_transform(y_pred)
    return jsonify({"predicted_city_tier": city_tier[0]})

@app.route("/predict/predict_improved_dependents", methods=["POST"])
def predict_improved_dependents():
    try:
        data = request.json
        df = pd.DataFrame([data])
        for col in ['Occupation', 'City_Tier']:
            df[col] = df[col].astype(str)
        df['edu_ratio'] = df['Education'] / (df['Income'] + 1)
        df['health_ratio'] = df['Healthcare'] / (df['Income'] + 1)
        df['grocery_ratio'] = df['Groceries'] / (df['Income'] + 1)
        df['entertainment_ratio'] = df['Entertainment'] / (df['Income'] + 1)
        df['total_spending_ratio'] = (
            df['Groceries'] + df['Education'] + df['Healthcare'] + df['Utilities']
        ) / (df['Income'] + 1)
        df['loan_ratio'] = df['Loan_Repayment'] / (df['Income'] + 1)
        df['savings_ratio'] = df['Income'] - (
            df['Rent'] + df['Loan_Repayment'] + df['Insurance'] +
            df['Groceries'] + df['Transport'] + df['Education'] +
            df['Healthcare'] + df['Entertainment'] + df['Utilities'] +
            df['Miscellaneous']
        )
        df['expense_ratio'] = (
            df['Groceries'] + df['Education'] + df['Healthcare']
        ) / (df['Income'] + 1)
        
        
        X_encoded = preprocessor.transform(df)
        y_pred = model.predict(X_encoded)
        dependents_pred = le.inverse_transform(y_pred)[0]
        
        return jsonify({"predicted_dependents": int(dependents_pred)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict/dependents", methods=["POST"])
def predict_dependents():
    data = request.json
    df = pd.DataFrame([data])
    for col in ['Occupation', 'City_Tier']:
        df[col] = df[col].astype(str)
    X_encoded = dependents_preprocessor.transform(df)
    y_pred = dependents_model.predict(X_encoded) + 1
    return jsonify({"predicted_dependents": int(y_pred[0])})


@app.route("/predict/financial_behavior", methods=["POST"])
def predict_financial_behavior():
    data = request.json
    df = pd.DataFrame([data])
    X_proc = financial_preprocessor.transform(df)
    preds = financial_model.predict(X_proc)[0]
    result = dict(zip(financial_target_cols, [float(x) for x in preds]))
    return jsonify(result)


@app.route("/predict/profession", methods=["POST"])
def predict_profession():
    data = request.json
    df = pd.DataFrame([data])
    X_encoded = profession_preprocessor.transform(df)
    pred_class = profession_model.predict(X_encoded)[0]
    pred_label = profession_labelencoder.inverse_transform([pred_class])[0]
    return jsonify({"predicted_profession": pred_label})


if __name__ == "__main__":
    app.run(debug=True)
