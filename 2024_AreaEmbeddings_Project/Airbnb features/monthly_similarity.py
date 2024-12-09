import pandas as pd
import numpy as np

# Path to your CSV file
input_file = 'quang/similarity_scores/llama3_no_listing_raw.csv'

# Read the data with proper column names
df = pd.read_csv(input_file, names=["group1_district", "group2_district", "month", "similarity_score"])

# Convert the 'Date' column to datetime and extract the month
df['month'] = pd.to_datetime(df['month'], errors='coerce').dt.to_period('M')

df['similarity_score'] = pd.to_numeric(df['similarity_score'], errors='coerce')

# Group by 'Month' and calculate statistics
results = df.groupby('month')['similarity_score'].agg(
    total_rows='count',
    mean='mean',
    median=lambda x: np.median(x),
    mode=lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    std_dev='std',
    min='min',
    max='max',
    third_quartile=lambda x: np.percentile(x, 75)  # 3분위수 추가
).reset_index()

# Save results to a CSV file
output_file = 'monthly_statistics.csv'
results.to_csv(output_file, index=False)

# Display the statistics
for index, row in results.iterrows():
    print(f"\nMonth: {row['month']}")
    print(f"Total Rows: {int(row['total_rows']):,}")
    print(f"Mean: {row['mean']:.6f}")
    print(f"Median: {row['median']:.6f}")
    print(f"Mode: {row['mode']:.6f}")
    print(f"Standard Deviation: {row['std_dev']:.6f}")
    print(f"Minimum: {row['min']:.6f}")
    print(f"Maximum: {row['max']:.6f}")
    print(f"3rd Quartile: {row['third_quartile']:.6f}")

print(f"\nResults have been saved to: {output_file}")
