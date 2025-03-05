import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (Replace 'medical_data.csv' with actual dataset)
data = pd.read_csv('medical_data.csv')

# Assume dataset has 'symptoms' as features and 'disease' as target
X = data.drop(columns=['disease'])
y = data['disease']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Function for user input-based prediction
def predict_disease(symptoms):
    input_data = np.array(symptoms).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Example usage
user_symptoms = [1, 0, 1, 0, 1]  # Example symptom input
predicted_disease = predict_disease(user_symptoms)
print(f'Predicted Disease: {predicted_disease}')
