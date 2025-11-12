import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import os
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("../datasets/purified_personal_datasets.csv")
df = df.dropna(subset=['Income', 'City_Tier', 'Occupation', 'Dependents'])

# Define input and target columns
input_cols = ['Income', 'City_Tier', 'Occupation', 'Dependents']
target_candidates = [col for col in df.select_dtypes(include=['int64','float64']).columns if col not in input_cols]
ignore_cols = ['Income','Dependents','Age']
target_cols = [col for col in target_candidates if col not in ignore_cols and not col.startswith("Potential_Savings_")]

# Preprocessing setup
cat_features = ['City_Tier', 'Occupation']
num_features = ['Income', 'Dependents']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# Prepare features and targets
X = df[input_cols]
Y = df[target_cols]
X_proc = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_proc, Y, test_size=0.2, random_state=42)

# Train multi-output XGBoost model
xgb = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
multi_model = MultiOutputRegressor(xgb)
multi_model.fit(X_train, Y_train)

# Evaluate
Y_pred = multi_model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
print(f"Overall MAE: {mae:.2f}, R2: {r2:.3f}")

Y_pred = multi_model.predict(X_test)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')

print(f"\n Overall Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# Evaluate per-column performance
r2_scores = {}
for i, col in enumerate(target_cols):
    r2_scores[col] = r2_score(Y_test.iloc[:, i], Y_pred[:, i])

r2_df = pd.DataFrame(r2_scores.items(), columns=["Feature", "R2_Score"]).sort_values("R2_Score", ascending=False)
print("\n Per-Feature R² Scores:")
print(r2_df)

# # Save model and preprocessor
# with open("../models/financial_behavior_model.pkl", "wb") as f:
#     pickle.dump(multi_model, f)
# with open("../models/financial_preprocessor.pkl", "wb") as f:
#     pickle.dump(preprocessor, f)

# print(" Model and preprocessor saved successfully!")

# Prediction function
def predict_financial_behavior(input_dict):
    with open("../models/financial_behavior_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../models/financial_preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    
    X_new = pd.DataFrame([input_dict])
    X_proc = preprocessor.transform(X_new)
    preds = model.predict(X_proc)[0]
    
    return dict(zip(target_cols, preds))

# Example usage
sample_input = {
    "Income": 85000,
    "City_Tier": "Tier 2",
    "Occupation": "Private Job",
    "Dependents": 2
}

predictions = predict_financial_behavior(sample_input)
print("\n Predicted Financial Behavior:")
for k, v in predictions.items():
    print(f"{k}: {v:.2f}")
