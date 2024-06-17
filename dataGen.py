import pandas as pd
import numpy as np

# Set the number of samples
num_samples = 1000

# Generate random data for features with realistic ranges
data = {
    'age': np.random.randint(20, 80, num_samples),
    'bmi': np.random.uniform(15, 40, num_samples),
    'bloodPressure': np.random.uniform(80, 180, num_samples),
    'totalCholesterol': np.random.uniform(150, 300, num_samples),
    'ldl': np.random.uniform(50, 200, num_samples),
    'hdl': np.random.uniform(30, 100, num_samples),
    'tsh': np.random.uniform(0.5, 5.0, num_samples),
    'lamotrigine': np.random.uniform(0, 10, num_samples),
    'bloodSugar': np.random.uniform(70, 150, num_samples),
}

# Create a balanced target column with more variability
data['target_column'] = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data.csv', index=False)
