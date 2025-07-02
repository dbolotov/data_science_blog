import os
import pandas as pd
import glob

# Set up directories relative to the script's location
base_dir = os.path.dirname(os.path.abspath(__file__))
extracted_dir = os.path.join(base_dir, "data", "extracted")
output_file = os.path.join(base_dir, "data", "cleaned_storm_data_ca.parquet")

# Columns to keep
columns_to_keep = [
    "BEGIN_DATE_TIME", "END_DATE_TIME",
    "EVENT_TYPE",
    "STATE", "CZ_NAME",
    "INJURIES_DIRECT", "INJURIES_INDIRECT",
    "DEATHS_DIRECT", "DEATHS_INDIRECT",
    "DAMAGE_PROPERTY", "DAMAGE_CROPS",
    "EVENT_NARRATIVE"
]

# List to collect dataframes
cleaned_dfs = []
original_total_rows = 0
filtered_total_rows = 0

# Process each extracted CSV file
for file in glob.glob(os.path.join(extracted_dir, "*.csv")):
    print(f"Processing: {file}")
    df = pd.read_csv(file, usecols=lambda col: col in columns_to_keep, dtype=str)

    original_total_rows += len(df)

    # Fill NA values
    df.fillna("0", inplace=True)

    # Filter for California only
    df = df[df['STATE'] == 'CALIFORNIA']

    # Filter rows with impact
    mask = ~((df['INJURIES_DIRECT'] == '0') &
             (df['INJURIES_INDIRECT'] == '0') &
             (df['DEATHS_DIRECT'] == '0') &
             (df['DEATHS_INDIRECT'] == '0') &
             (df['DAMAGE_PROPERTY'].isin(['0.00K', '0.00M', '0.00B', '0'])) &
             (df['DAMAGE_CROPS'].isin(['0.00K', '0.00M', '0.00B', '0'])))

    df = df[mask]
    filtered_total_rows += len(df)
    cleaned_dfs.append(df)

# Combine all filtered data
combined_df = pd.concat(cleaned_dfs, ignore_index=True)

# Save as Parquet
combined_df.to_parquet(output_file, index=False)

print("\nData cleaning complete.")
print(f"Original total rows: {original_total_rows}")
print(f"Rows after filtering for CA and impact: {filtered_total_rows}")
print(f"Saved cleaned data to: {output_file}")
