from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load dataset
data = load_diabetes()
X = data.data
y_raw = data.target

# Convert regression target to binary (for classification model)
# We'll assume a simple rule: high risk if target > 140
binarizer = Binarizer(threshold=140)
y = binarizer.fit_transform(y_raw.reshape(-1, 1)).ravel()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Save model to file
joblib.dump(model, 'models/diabetes_model.pkl')
print("✅ Model saved to models/diabetes_model.pkl")
