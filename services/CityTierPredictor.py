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
X = df[selected_cols].copy()
y = df['City_Tier']


le = LabelEncoder()
y = le.fit_transform(y)

cat_cols = ['Occupation']
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


pickle.dump(model, open("../models/city_tier_model.pkl", "wb"))
pickle.dump(preprocessor, open("../models/city_tier_preprocessor.pkl", "wb"))
pickle.dump(le, open("../models/city_tier_labelencoder.pkl", "wb"))

print("\n✅ Model and preprocessors saved for deployment!")
