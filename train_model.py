import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# -----------------------------
# LOAD DATA (header row = 1)
# -----------------------------

data = pd.read_excel(
    "MARK Survey Data_Mother Sheet_17th July 2018_pharmEDGE_1.xlsx",
    header=1
)

print("Columns Loaded:")
print(data.columns.tolist())

# -----------------------------
# SELECT FEATURES
# -----------------------------

features = [
    "Age\n(years)",
    "Blood pressure treatment",
    "Smoking status",
    "Angina or heart attack",
    "Systolic blood pressure\n(mmHg)",
    "Total cholesterol/HDL ratio",
    "BMI (kg/m2)",
    "Gender",
    "Chronic kidney disease",
    "HDL-cholesterol\n(mg/dL)",
    "Total cholesterol\n(mg/dL)",
    "Atrial fibrillation"
]

target = "10-years QRISK3 score (%)"

# Remove rows with missing values
data = data[features + [target]].dropna()

X = data[features]
y = data[target]

# -----------------------------
# TRUE CATEGORICAL VARIABLES
# -----------------------------

cat_features = [
    "Blood pressure treatment",
    "Smoking status",
    "Angina or heart attack",
    "Gender",
    "Chronic kidney disease",
    "Atrial fibrillation"
]

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN REGRESSION MODEL
# -----------------------------

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# -----------------------------
# EVALUATE MODEL
# -----------------------------

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# -----------------------------
# SAVE MODEL
# -----------------------------

pickle.dump(model, open("catboost_corrected.pickle", "wb"))

print("Model saved as catboost_corrected.pickle")