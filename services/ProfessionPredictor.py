import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv("../datasets/purified_personal_datasets.csv")

target_col = 'Occupation'

feature_cols = [
    'Income', 'Age', 'City_Tier',
    'Rent', 'Loan_Repayment', 'Insurance',
    'Groceries', 'Transport', 'Education',
    'Healthcare', 'Entertainment', 'Utilities', 'Miscellaneous',
    'Dependents'
]

X = df[feature_cols].copy()
y = df[target_col].copy()

le = LabelEncoder()
y = le.fit_transform(y)

cat_cols = ['City_Tier', 'Dependents']  # categorical columns
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

X_encoded = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

num_classes = len(np.unique(y))

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='multi:softmax',
    num_class=num_classes,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)

print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

pickle.dump(model, open("profession_model.pkl", "wb"))
pickle.dump(preprocessor, open("profession_preprocessor.pkl", "wb"))
pickle.dump(le, open("profession_labelencoder.pkl", "wb"))

print("\n✅ Profession Predictor model saved successfully!")
