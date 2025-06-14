# models/train_heart_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
df = pd.read_csv("../heart_cleveland_upload.csv")

# Preprocess if needed (assuming all fields are numeric and clean)
X = df.drop('condition', axis=1)
y = df['condition']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Heart Model Accuracy: {round(acc, 2)}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/heart_model.pkl")
print("✅ Model saved to models/heart_model.pkl")
