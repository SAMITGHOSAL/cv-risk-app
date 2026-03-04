import pickle
import pandas as pd

# Load model
model = pickle.load(open("catboost.pickle", "rb"))

# Example patient input (adjust categories exactly as used in training)
sample_input = {
    "Age": ">45",
    "Blood pressure treatment": "YES",
    "Smoking status": "Non-smoker",
    "Angina or heart attack": "NO",
    "Systolic blood pressure": ">=130",
    "Total cholesterol/HDL ratio": ">4",
    "BMI": "Overweight",
    "Gender": "Male",
    "Chronic kidney disease": "NO",
    "Total cholesterol": ">200",
    "HDL-cholesterol": "<40",
    "Atrial fibrillation": "NO"
}

df = pd.DataFrame([sample_input])

prediction = model.predict(df)
probability = model.predict_proba(df)

print("Prediction:", prediction)
print("Probability:", probability)