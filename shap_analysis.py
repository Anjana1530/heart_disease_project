import shap
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load data and model
df = pd.read_csv("../heart.csv")
print(df.columns)

# Separate features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Scale the features
scaler = joblib.load("../model/scaler.pkl")
X_scaled = scaler.transform(X)

# Load model
model = load_model("../model/heart_model.h5")

# Explain predictions with SHAP
explainer = shap.Explainer(model, X_scaled)
shap_values = explainer(X_scaled[:10])

# Show summary plot
shap.summary_plot(shap_values, X.iloc[:10], feature_names=X.columns)

  # Use [0] for class 0 explanation

