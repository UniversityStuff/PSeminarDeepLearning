"""
Script to combine two CSV files:
- Takes the base data from 'landslides_Rain_SM_with_slope_AntiGravity.csv'
- Replaces only the rain columns with data from 'landslides_WITH_NEW_RAIN.csv'
- Outputs to 'landslides_beta.csv'
"""

import pandas as pd
import os

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
original_file = os.path.join(BASE_DIR, "landslides_Rain_SM_with_slope_AntiGravity.csv")
new_rain_file = os.path.join(BASE_DIR, "landslides_WITH_NEW_RAIN.csv")
output_file = os.path.join(BASE_DIR, "landslides_beta.csv")

# Rain columns to replace
RAIN_COLUMNS = ['Rainfall_7d_mm', 'Rainfall_14d_mm', 'Rainfall_21d_mm']

def main():
    print("Loading original data...")
    df_original = pd.read_csv(original_file)
    print(f"  Original shape: {df_original.shape}")
    
    print("Loading new rain data...")
    df_new_rain = pd.read_csv(new_rain_file)
    print(f"  New rain data shape: {df_new_rain.shape}")
    
    # Drop the old rain columns from the original dataframe
    df_result = df_original.drop(columns=RAIN_COLUMNS)
    
    # Get only UUID and rain columns from the new rain file
    df_rain_only = df_new_rain[['UUID'] + RAIN_COLUMNS]
    
    # Merge the new rain data using UUID as the key
    df_result = df_result.merge(df_rain_only, on='UUID', how='left')
    
    # Reorder columns to match original order
    original_columns = df_original.columns.tolist()
    df_result = df_result[original_columns]
    
    print(f"  Result shape: {df_result.shape}")
    
    # Check for any missing rain data after merge
    missing_rain = df_result[RAIN_COLUMNS].isna().any(axis=1).sum()
    if missing_rain > 0:
        print(f"  Warning: {missing_rain} rows have missing rain data after merge")
    
    # Save to output file
    df_result.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved combined data to: {output_file}")

if __name__ == "__main__":
    main()
