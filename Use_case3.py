import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

claims = pd.read_excel("Dataset.xlsx", sheet_name="Claims")
dealers = pd.read_excel("Dataset.xlsx", sheet_name="Dealers")

claims["Vehicle_Age"] = np.random.randint(1, 15, size=len(claims))
claims["Mileage"] = np.random.randint(10000, 200000, size=len(claims))
claims["Repair_or_Replace"] = np.random.choice([0, 1], size=len(claims))

claims = pd.merge(claims, dealers[["Dealer_ID", "State"]], on="Dealer_ID", how="left")

features = claims[["Vehicle_Age", "Mileage", "State"]]
target = claims["Repair_or_Replace"]

features_encoded = pd.get_dummies(features, columns=["State"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=features_encoded.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))
