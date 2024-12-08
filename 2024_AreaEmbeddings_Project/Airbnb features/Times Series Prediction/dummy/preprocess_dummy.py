import pandas as pd
import numpy as np

# Load the dataframes
df1 = pd.read_csv("Dataset/Raw_Embeddings/Road_Embeddings.csv")
df2 = pd.read_csv("Dataset/Raw_Embeddings/HumanFlow_Embeddings_kor.csv")
df3 = pd.read_csv("Dataset/AirBnB_LLM/gemma2_no_listing_refined_fill_na_ver.csv")

# Step 1: Prepare DataFrame 1 (expand for all 67 months)
unique_months = df3["Reporting Month"].unique()
unique_months.sort()  # Ensure chronological order
expanded_df1 = df1.loc[df1.index.repeat(len(unique_months))].reset_index(drop=True)
expanded_df1["Reporting Month"] = np.tile(unique_months, len(df1))

# Rename 'ADM_NM' to 'ADM_NM_key' for consistency
expanded_df1.rename(columns={"ADM_NM": "ADM_NM_key"}, inplace=True)

# Step 2: Align `기준일ID` in DataFrame 2 with `Reporting Month` in DataFrame 3
df2["Reporting Month"] = pd.to_datetime(df2["기준일ID"], format='%Y%m').dt.strftime('%Y-%m-%d')

# Rename columns for consistency
df2.rename(columns={"행정동코드": "ADM_NM_key"}, inplace=True)
df3.rename(columns={"ADM_NM": "ADM_NM_key"}, inplace=True)

# Step 3: Create a Cartesian product of ADM_NM_key and Reporting Month to ensure all combinations
adm_nm_keys = df1["ADM_NM"].unique()
cartesian_product = pd.DataFrame(
    [(adm, month) for month in unique_months for adm in adm_nm_keys],
    columns=["ADM_NM_key", "Reporting Month"]
)

# Merge DataFrames to ensure all ADM_NM_key and Reporting Month pairs are included
merged_df = pd.merge(cartesian_product, expanded_df1, on=["ADM_NM_key", "Reporting Month"], how="left")
merged_df = pd.merge(merged_df, df2, on=["ADM_NM_key", "Reporting Month"], how="left")
merged_df = pd.merge(merged_df, df3, on=["ADM_NM_key", "Reporting Month"], how="left")

# Step 4: Prepare embeddings
embedding_cols_df1 = [col for col in df1.columns if col != "ADM_NM"]  # Maintain original order from df1
embedding_cols_df2 = [col for col in df2.columns if col not in ["ADM_NM_key", "Reporting Month", "기준일ID"]]  # Original order from df2
embedding_cols_df3 = ["LLM Embeddings"]  # Only 1 column from df3

# Combine embeddings: handle all but the last column
embeddings = pd.concat([
    merged_df[embedding_cols_df1],  # 23 columns from df1
    merged_df[embedding_cols_df2],  # 35 columns from df2
    merged_df[embedding_cols_df3]   # Keep LLM Embeddings as-is
], axis=1)

# Step 5: Prepare labels
labels = merged_df[["Occupancy Rate", "Revenue (USD)", "Number of Reservations"]]

# Sort the data: ADM_NM within each month, and months in chronological order
adm_nm_order = df1["ADM_NM"].tolist()
merged_df_sorted = merged_df.sort_values(
    by=["Reporting Month", "ADM_NM_key"],
    key=lambda col: pd.Categorical(col, categories=adm_nm_order, ordered=True) if col.name == "ADM_NM_key" else pd.to_datetime(col)
)

# Separate embeddings and labels
embeddings_sorted = embeddings.loc[merged_df_sorted.index]
labels_sorted = labels.loc[merged_df_sorted.index]

# Step 6: Handle missing values and ensure float conversion for all but the last column
# Exclude the last column ("LLM Embeddings") from processing
embeddings_without_last = embeddings_sorted.iloc[:, :-1].fillna(0.0).astype(float)
embeddings_last_column = embeddings_sorted.iloc[:, -1]  # Preserve "LLM Embeddings" as-is

# Concatenate the processed embeddings with the last column
embeddings_final = pd.concat([embeddings_without_last, embeddings_last_column], axis=1)

# Handle missing values and ensure float for labels
labels_sorted = labels_sorted.fillna(0.0).astype(float)

# Validate row count
assert embeddings_final.shape[0] == 28408, "Row count for embeddings is incorrect!"
assert labels_sorted.shape[0] == 28408, "Row count for labels is incorrect!"

# Validate column count
assert embeddings_final.shape[1] == 59, "Column count for embeddings is incorrect!"

# Step 7: Save to CSV
embeddings_final.to_csv("embeddings.csv", index=False)
labels_sorted.to_csv("labels.csv", index=False)
