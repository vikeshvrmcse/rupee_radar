import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle

df=pd.read_csv("../datasets/Indian Personal Finance and Spending Habits.csv")
newdf=df.iloc[:,:19]
newdf = newdf[(newdf != 0).all(axis=1)]
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
outdf=remove_outliers_iqr(newdf, 'Income')
copydf=outdf.copy()
all_features = pd.get_dummies(copydf, columns=['Occupation', 'City_Tier'], drop_first=True)

A_input=all_features.drop(columns=['Income','Disposable_Income','Desired_Savings_Percentage'])
B_output=all_features['Income']

# Split data
A_X_train, A_X_test, B_Y_train, B_Y_test = train_test_split(
    A_input, B_output, test_size=0.3, random_state=62
)

# Train model
modelR = RandomForestRegressor(random_state=42)
modelR.fit(A_X_train, B_Y_train)

# Predict
A_y_pred = modelR.predict(A_X_test)

# Evaluate
print("Mean Squared Error: ", mean_squared_error(B_Y_test, A_y_pred))
print("R2 Score: ", r2_score(B_Y_test, A_y_pred))

# Cross-validation
# cv_scores = cross_val_score(modelR, A_input, B_output, cv=5)
# print("Cross Validation Score: ", cv_scores)
# print("Average CV Score: ", cv_scores.mean())



# Example test data
new_data = pd.DataFrame({
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

# Recreate dummies for new_data
new_data_encoded = pd.get_dummies(new_data, columns=['Occupation', 'City_Tier'], drop_first=True)

# Align new data columns with training data columns
new_data_encoded = new_data_encoded.reindex(columns=A_X_train.columns, fill_value=0)

# Predict
prediction = modelR.predict(new_data_encoded)
print("Predicted Income:", prediction[0])

with open("../models/income_model.pkl", "wb") as f:
    pickle.dump(modelR, f)

print("💾 Model saved successfully as '../models/income_model.pkl'")
