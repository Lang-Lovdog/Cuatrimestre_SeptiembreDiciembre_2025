import pandas as pd

# Read both CSV files
complete_results = pd.read_csv('complete_test_results.csv')
summary_best = pd.read_csv('summary_best_models.csv')

# Get the unique combinations of Model, Preprocessing, and Columns_Dropped from summary
summary_combinations = summary_best[['Model', 'Preprocessing', 'Columns_Dropped']].drop_duplicates()

# Filter complete_results to find records that match any of these combinations
matching_records = pd.merge(
    complete_results,
    summary_combinations,
    on=['Model', 'Preprocessing', 'Columns_Dropped'],
    how='inner'
)

# Display the results
print("Matching records found:")
print(f"Number of matches: {len(matching_records)}")
print(f"Unique combinations: {len(summary_combinations)}")

# Export separate CSV files for each combination
for _, combo in summary_combinations.iterrows():
    model = combo['Model']
    preprocessing = combo['Preprocessing']
    columns_dropped = combo['Columns_Dropped']
    
    # Filter records for this specific combination
    combo_records = matching_records[
        (matching_records['Model'] == model) &
        (matching_records['Preprocessing'] == preprocessing) &
        (matching_records['Columns_Dropped'] == columns_dropped)
    ]
    
    # Create filename based on the pattern
    drop_suffix = "_drop" if columns_dropped else ""
    filename = f"{model}_{preprocessing}{drop_suffix}_results.csv"
    
    # Clean filename of any invalid characters
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
    
    # Save to CSV
    combo_records.to_csv(filename, index=False)
    print(f"Exported: {filename} ({len(combo_records)} records)")

print(f"\nTotal: {len(summary_combinations)} files exported")

# Also save all matching records together for reference
matching_records.to_csv('all_matching_records.csv', index=False)
print("\nAll matching records also saved to 'all_matching_records.csv'")
