"""
train_model.py
Trains a Logistic Regression model on the actual diabetes.csv dataset
(Pima Indians Diabetes dataset) using all 8 available features, and
saves a full preprocessing + model pipeline to model.pkl for deployment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

# ---------------------------------------------------------
# 1. Load the real dataset (NOT a hardcoded toy sample)
# ---------------------------------------------------------
df = pd.read_csv("diabetes.csv")

# ---------------------------------------------------------
# 2. Data preprocessing
#    Several columns use 0 as a placeholder for a missing
#    reading (a person cannot have 0 BMI or 0 Glucose), so
#    we treat those zeros as missing values and impute them
#    with the column median.
# ---------------------------------------------------------
cols_with_zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_with_zero_as_missing:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# No categorical columns are present in this dataset, so no
# encoding step is required here.

# ---------------------------------------------------------
# 3. Feature / target split (ALL 8 features are used)
# ---------------------------------------------------------
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

X = df[FEATURE_NAMES]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4. Build model as a Pipeline (scaler + classifier) so the
#    deployed app can hand it raw feature values directly.
# ---------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

pipeline.fit(X_train, y_train)

print("Train accuracy:", pipeline.score(X_train, y_train))
print("Test accuracy:", pipeline.score(X_test, y_test))

# ---------------------------------------------------------
# 5. Save the pipeline AND the feature name order together,
#    so app.py always knows what to ask for and in what order.
# ---------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump({"pipeline": pipeline, "feature_names": FEATURE_NAMES}, f)

print("Saved model.pkl with features:", FEATURE_NAMES)
print("DONE")
