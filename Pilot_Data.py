import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv('Pilot_Performance_Dataset.csv')

# 1. Data Preprocessing
# Convert categorical variables to numeric using LabelEncoder
label_encoders = {}
for column in ['Scenario', 'Cockpit Item', 'Correct Action?']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target
X = data.drop(columns=['Pilot ID', 'Correct Action?'])
y = data['Correct Action?']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 5. Feature Importance
feature_importance = pd.DataFrame({'Feature': data.drop(columns=['Pilot ID', 'Correct Action?']).columns,
                                    'Importance': model.feature_importances_})
feature_importance.sort_values(by='Importance', ascending=False, inplace=True)
print("\nFeature Importance:\n", feature_importance)

# Save the feature importance for analysis
feature_importance.to_csv('feature_importance.csv', index=False)

# 6. Save preprocessed data for future use
processed_data = pd.DataFrame(X_scaled, columns=data.drop(columns=['Pilot ID', 'Correct Action?']).columns)
processed_data['Correct Action?'] = y
processed_data.to_csv('processed_pilot_data.csv', index=False)

print("Data preprocessing, model training, and evaluation completed.")
