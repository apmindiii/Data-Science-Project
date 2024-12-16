import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

claims = pd.read_excel("Dataset.xlsx", sheet_name="Claims")

claims = {
    "Dealer_ID": np.random.randint(1, 101, 500),
    "Customer_Satisfaction_Score": np.random.randint(1, 6, 500),  # Score: 1-5
    "Churn_Rate": np.random.uniform(0, 1, 500),  # Churn rate: 0 to 1
    "Preference_Change": np.random.choice([0, 1], 500),  # 0: No change, 1: Changed
    "Feedback_Score": np.random.randint(1, 11, 500),  # Feedback: 1-10
    "Retention_Status": np.random.choice([0, 1], 500),  # 0: Not Retained, 1: Retained
}

customer_data = pd.DataFrame(claims)

features = customer_data[
    ["Customer_Satisfaction_Score", "Churn_Rate", "Preference_Change", "Feedback_Score"]
]
target = customer_data["Retention_Status"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=features.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

print("\nInsights:")
print(f"- Average Customer Satisfaction Score: {customer_data['Customer_Satisfaction_Score'].mean():.2f}")
print(f"- Average Feedback Score: {customer_data['Feedback_Score'].mean():.2f}")
print(f"- Retention Rate: {customer_data['Retention_Status'].mean() * 100:.2f}%")
