import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
X = []
y = []

for file in os.listdir("dataset"):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        with open(os.path.join("dataset", file)) as f:
            reader = csv.reader(f)
            for row in reader:
                X.append([float(val) for val in row])
                y.append(label)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save model and label encoder
joblib.dump(model, "model/isl_gesture_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")
