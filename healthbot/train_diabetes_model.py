import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the dataset
df = pd.read_csv("diabetes.csv")  # Update path if needed

# Clean missing data
df.dropna(inplace=True)

# Feature set and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Diabetes Model Accuracy: {accuracy:.2f}")

# Save the model to a .pkl file
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/diabetes_model.pkl")
print("✅ Model saved to models/diabetes_model.pkl")