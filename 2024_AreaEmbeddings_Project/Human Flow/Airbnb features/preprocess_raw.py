import pandas as pd
import numpy as np
import ast

# Load the dataframes
df1 = pd.read_csv("Dataset/Raw_Embeddings/Road_Embeddings.csv")
df2 = pd.read_csv("Dataset/Raw_Embeddings/HumanFlow_Embeddings_kor.csv")
df3 = pd.read_csv("Dataset/AirBnB_LLM/llama3_mixed_raw_fill_na_ver.csv")

# Step 1: Prepare DataFrame 1 (expand for all 67 months)
unique_months = df3["Reporting Month"].unique()
unique_months.sort()  # Ensure chronological order
expanded_df1 = df1.loc[df1.index.repeat(len(unique_months))].reset_index(drop=True)
expanded_df1["Reporting Month"] = np.tile(unique_months, len(df1))
expanded_df1.rename(columns={"ADM_NM": "ADM_NM_key"}, inplace=True)

# Step 2: Align `기준일ID` in DataFrame 2 with `Reporting Month` in DataFrame 3
df2["Reporting Month"] = pd.to_datetime(df2["기준일ID"], format='%Y%m').dt.strftime('%Y-%m-%d')
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

# Debug: Check merged_df shape
print(f"Merged DataFrame shape: {merged_df.shape}")

# Step 4: Prepare embeddings
embedding_cols_df1 = [col for col in df1.columns if col != "ADM_NM"]  # Maintain original order from df1
embedding_cols_df2 = [col for col in df2.columns if col not in ["ADM_NM_key", "Reporting Month", "기준일ID"]]  # Original order from df2
embedding_cols_df3 = ["LLM Embeddings"]  # Only 1 column from df3

# Combine embeddings: handle all but the last column
embeddings = pd.concat([
    merged_df[embedding_cols_df1],  # 23 columns from df1
    merged_df[embedding_cols_df2],  # 35 columns from df2
    merged_df[embedding_cols_df3]   # Keep LLM Embeddings as a string for now
], axis=1)

# Debug: Check embeddings shape
print(f"Embeddings DataFrame shape before flattening: {embeddings.shape}")

# Step 5: Parse and Flatten the last column (LLM Embeddings)
def parse_llm_embeddings(value):
    try:
        return [float(v) for v in ast.literal_eval(value)]
    except (ValueError, SyntaxError, TypeError):
        return [0.0] * 2304

llm_embeddings_flat = embeddings["LLM Embeddings"].apply(parse_llm_embeddings)

# Convert the flattened embeddings into a NumPy matrix
llm_embeddings_matrix = np.array(llm_embeddings_flat.tolist())

# Debug: Check LLM Embeddings matrix shape
print(f"LLM Embeddings matrix shape: {llm_embeddings_matrix.shape}")

# Combine all columns into a single matrix
embeddings_without_last = embeddings.iloc[:, :-1].fillna(0.0).astype(float).values  # First 58 columns as a NumPy array
final_embeddings_matrix = np.hstack((embeddings_without_last, llm_embeddings_matrix))

# Debug: Check final_embeddings_matrix shape
print(f"Final embeddings matrix shape: {final_embeddings_matrix.shape}")

# Prepare labels
labels = merged_df[["Occupancy Rate", "Revenue (USD)", "Number of Reservations"]].fillna(0.0).astype(float).values

# Debug: Check labels shape
print(f"Labels array shape: {labels.shape}")

# Step 6: Save as CSV
np.savetxt("quang/time_series_prediction/raw_embeddings/llama_mixed_raw_ver.csv", final_embeddings_matrix, delimiter=",", fmt="%.8f")  # Save embeddings
# np.savetxt("labels.csv", labels, delimiter=",", fmt="%.8f")  # Save labels

# Check for row mismatch
if final_embeddings_matrix.shape[0] != labels.shape[0]:
    print(f"Row mismatch detected: embeddings have {final_embeddings_matrix.shape[0]} rows, labels have {labels.shape[0]} rows.")
else:
    print("Embeddings and labels row counts match.")
