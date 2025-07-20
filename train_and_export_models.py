# train_and_export_models.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Load datasets
dataset1 = pd.read_csv("Dataset_1.csv")
dataset2 = pd.read_csv("Dataset_2.csv")

# Split features and labels
X1 = dataset1.drop(columns=["target"])
y1 = dataset1["target"]
X2 = dataset2.drop(columns=["Heart_Risk"])
y2 = dataset2["Heart_Risk"]

# Train models
model1 = GaussianNB()
model1.fit(X1, y1)

model2 = GradientBoostingClassifier()
model2.fit(X2, y2)

# Save models and feature lists
joblib.dump(model1, "model1_naive_bayes.pkl")
joblib.dump(model2, "model2_gb.pkl")
joblib.dump(list(X1.columns), "model1_features.pkl")
joblib.dump(list(X2.columns), "model2_features.pkl")

print("✅ Models and features exported successfully.")
