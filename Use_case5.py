import pandas as pd

file_path = 'Dataset.xlsx'

claims_data = pd.read_excel(file_path, sheet_name="Claims")
dealers_data = pd.read_excel(file_path, sheet_name="Dealers")
parts_data = pd.read_excel(file_path, sheet_name="Parts")

claims_data_cleaned = claims_data.drop_duplicates()

claims_combined = claims_data_cleaned.merge(dealers_data, how='left', on='Dealer_ID')
claims_combined = claims_combined.merge(parts_data, how='left', on='Part_ID')

claims_combined.fillna({
    'Dealer_Name': 'Unknown Dealer',
    'Part_Name': 'Unknown Part'
}, inplace=True)

top_dealers = claims_combined.groupby('Dealer_Name')['claim_amount'].sum().sort_values(ascending=False).head(5)

top_parts = claims_combined['Part_Name'].value_counts().head(5)

print("Top 5 Dealers by Claim Amount:")
print(top_dealers)

print("\nTop 5 Most Frequently Claimed Parts:")
print(top_parts)

processed_file = 'Processed_Claims_Data.xlsx'
claims_combined.to_excel(processed_file, index=False)
print(f"\nProcessed data saved to {processed_file}")
