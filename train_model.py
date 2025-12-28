import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("insurance.csv")  # make sure file exists

X = df.drop(columns="charges")
y = np.log(df["charges"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

# Pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "insurance_model.pkl")

print("✅ Model trained and saved successfully")
