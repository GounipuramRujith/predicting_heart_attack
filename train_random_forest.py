# train_random_forest.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib # Import joblib to save the model

# Adjust this path if your CSV is in a different location
df = pd.read_csv("/Users/rujith/Downloads/merged_heart_attack_dataset.csv") # <--- IMPORTANT: Update this path!

x = df.drop("heart_attack_risk", axis=1)
y = df["heart_attack_risk"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25, random_state=42) # Added random_state for reproducibility

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(x_train, y_train)
y_predict = random_forest.predict(x_test) # Predict on x_test, not y_test

print("Random Forest Model Performance:")
print("Accuracy Score:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Save the trained model
joblib.dump(random_forest, 'random_forest_model.pkl')
print("Random Forest model saved as random_forest_model.pkl")
