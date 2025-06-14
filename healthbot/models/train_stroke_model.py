import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("../healthcare-dataset-stroke-data.csv")

# Preprocess
df.dropna(inplace=True)
df.replace({
    'gender': {'Male': 0, 'Female': 1, 'Other': 2},
    'ever_married': {'No': 0, 'Yes': 1},
    'work_type': {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4},
    'Residence_type': {'Rural': 0, 'Urban': 1},
    'smoking_status': {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
}, inplace=True)

# Features and Target
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Stroke Model Accuracy: {accuracy:.2f}")

# Save model
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stroke_model.pkl")
print("✅ Model saved to models/stroke_model.pkl")
