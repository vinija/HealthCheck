import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Generate synthetic dataset for demonstration
data = pd.DataFrame({
    'age': [25, 35, 45, 55, 65, 75, 85, 95, 105, 115],
    'bmi': [22, 24, 26, 28, 30, 32, 34, 36, 38, 40],
    'bloodPressure': [120, 124, 128, 132, 136, 140, 144, 148, 152, 156],
    'totalCholesterol': [180, 185, 190, 195, 200, 205, 210, 215, 220, 225],
    'ldl': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
    'hdl': [60, 62, 64, 66, 68, 70, 72, 74, 76, 78],
    'tsh': [2, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8],
    'lamotrigine': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'bloodSugar': [90, 95, 100, 105, 110, 115, 120, 125, 130, 135],
    'risk': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]  # Assume these are the correct labels
})

X = data.drop('risk', axis=1)
y = data['risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and the scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
