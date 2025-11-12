# ============================================
# 🧠 Train Dependents Prediction Model
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("../datasets/purified_personal_datasets.csv")

# -------------------------------
# 2️⃣ Feature engineering
# -------------------------------
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

# -------------------------------
# 3️⃣ Select features & target
# -------------------------------

selected_cols = [
    'Income', 'Age', 'Occupation', 'City_Tier',
    'Rent', 'Loan_Repayment', 'Insurance',
    'Groceries', 'Transport', 'Education',
    'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous',
    'edu_ratio', 'health_ratio', 'grocery_ratio',
    'entertainment_ratio', 'total_spending_ratio',
    'loan_ratio', 'savings_ratio', 'expense_ratio'
]
X = df[selected_cols].copy()

# Encode target labels to start from 0
le = LabelEncoder()
y = le.fit_transform(df['Dependents'])

# -------------------------------
# 4️⃣ Preprocessing setup
# -------------------------------
cat_cols = ['Occupation', 'City_Tier']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Transform features
X_encoded = preprocessor.fit_transform(X)

# -------------------------------
# 5️⃣ Train-test split
# -------------------------------


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6️⃣ Apply SMOTE for balancing
# -------------------------------


sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------------------
# 7️⃣ Train XGBoost model
# -------------------------------


num_classes = len(np.unique(y_train_res))

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=num_classes,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train_res, y_train_res)

# -------------------------------
# 8️⃣ Save model, preprocessor, and label encoder
# -------------------------------

with open("../models/dependents_improved_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../models/dependents_improved_preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("../models/dependents_improved_labelencoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model, preprocessor, and label encoder saved successfully!")
