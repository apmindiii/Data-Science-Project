import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

claims = pd.read_excel("Dataset.xlsx", sheet_name="Claims")
dealers = pd.read_excel("Dataset.xlsx", sheet_name="Dealers")
parts = pd.read_excel("Dataset.xlsx", sheet_name="Parts")

claims_dealers = pd.merge(claims, dealers, on="Dealer_ID")

claims_parts = pd.merge(claims, parts, on="Part_ID")

top_10_dealers = claims_dealers["Dealer_Name"].value_counts().head(10)

top_10_failed_parts = claims_parts["Part_Name"].value_counts().head(10)

print("Top 10 Dealers:\n", top_10_dealers)
print("\nTop 10 Failed Parts:\n", top_10_failed_parts)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_10_dealers.index, y=top_10_dealers.values)
plt.title("Top 10 Dealers by Warranty Claims")
plt.xlabel("Dealer")
plt.ylabel("Name of Customer")
plt.xticks(rotation=25)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=top_10_failed_parts.index, y=top_10_failed_parts.values)
plt.title("Top 10 Failed Parts")
plt.xlabel("Part")
plt.ylabel("Number of Claims")
plt.xticks(rotation=25)
plt.show()
