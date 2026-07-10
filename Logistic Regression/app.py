import streamlit as st
import pickle
import numpy as np
import os

# ---------------------------------------------------------
# Load the trained pipeline (scaler + logistic regression)
# and the feature name order it expects.
# ---------------------------------------------------------
model_path = os.path.join(os.getcwd(), "model.pkl")
with open(model_path, "rb") as f:
    saved = pickle.load(f)

pipeline = saved["pipeline"]
feature_names = saved["feature_names"]

st.title("Diabetes Risk Prediction App")
st.write(
    "Enter the patient's health measurements below to predict the "
    "likelihood of diabetes (Outcome: 1 = Diabetic, 0 = Not Diabetic)."
)

# ---------------------------------------------------------
# Inputs — ALL 8 features the model was trained on
# ---------------------------------------------------------
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)
age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

if st.button("Predict"):
    # Build the input in the exact same feature order used during training
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]])

    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"High risk of Diabetes (probability: {probability:.2%})")
    else:
        st.success(f"Low risk of Diabetes (probability: {probability:.2%})")
