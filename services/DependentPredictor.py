import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("../datasets/purified_personal_datasets.csv")

# -------------------------------
# 2️⃣ Feature & target selection (original dataset columns only)
# -------------------------------
feature_cols = [
    'Income', 'Age', 'Occupation', 'City_Tier',
    'Rent', 'Loan_Repayment', 'Insurance',
    'Groceries', 'Transport', 'Education',
    'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous'
]

X = df[feature_cols].copy()

# -------------------------------
# 3️⃣ Target variable (Dependents)
# -------------------------------
y = df['Dependents'] - 1  # Convert labels 1–5 → 0–4

# -------------------------------
# 4️⃣ Preprocessing setup
# -------------------------------
cat_cols = ['Occupation', 'City_Tier']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

X_encoded = preprocessor.fit_transform(X)

# -------------------------------
# 5️⃣ Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6️⃣ Apply SMOTE to handle imbalance
# -------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# -------------------------------
# 7️⃣ Train XGBoost model
# -------------------------------
num_classes = len(np.unique(y))
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

y_pred = model.predict(X_test)

y_test_shifted = y_test + 1
y_pred_shifted = y_pred + 1

print("\n✅ Model Accuracy:", round(accuracy_score(y_test_shifted, y_pred_shifted), 4))
print("\n📊 Classification Report:\n")
print(classification_report(
    y_test_shifted, y_pred_shifted,
    digits=3,
    target_names=[f"Dependents_{i}" for i in range(1, len(np.unique(y_test_shifted)) + 1)]
))

print("\n🔍 Sample Predictions (first 10):")
for true, pred in zip(y_test_shifted[:10], y_pred_shifted[:10]):
    print(f"Actual: {true} → Predicted: {pred}")

# -------------------------------
# 8️⃣ Save model and preprocessor
# -------------------------------
os.makedirs("./models", exist_ok=True)

with open("./models/dependents_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("./models/dependents_preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("\n💾 Model and preprocessor saved successfully as .pkl files!")
