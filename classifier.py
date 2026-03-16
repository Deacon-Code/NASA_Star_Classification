import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


train = pd.read_csv("train.csv")

color_encoder    = LabelEncoder()
spectral_encoder = LabelEncoder()

train["Color_enc"]    = color_encoder.fit_transform(train["Color"])
train["Spectral_enc"] = spectral_encoder.fit_transform(train["Spectral_Class"])

FEATURE_COLS = ["Temperature", "L", "R", "A_M", "Color_enc", "Spectral_enc"]


X_train = train[FEATURE_COLS].values   
y_train = train["Type"].values          


scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_train)

print("=== Confusion Matrix (rows=actual, cols=predicted) ===")
print(confusion_matrix(y_train, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_train, y_pred,
                             target_names=[f"Type {i}" for i in range(6)]))


example_raw = pd.DataFrame([{
    "Temperature": 3068,
    "L":           0.0024,
    "R":           0.17,
    "A_M":         16.12,
    "Color_enc":   color_encoder.transform(["Red"])[0],
    "Spectral_enc": spectral_encoder.transform(["M"])[0],
}])

example_scaled      = scaler.transform(example_raw[FEATURE_COLS].values)
predicted_type      = model.predict(example_scaled)[0]
predicted_probs     = model.predict_proba(example_scaled)[0]

print(f"\n=== Single-star prediction ===")
print(f"Predicted star type : {predicted_type}")
print(f"Class probabilities : { {i: round(p, 3) for i, p in enumerate(predicted_probs)} }")






