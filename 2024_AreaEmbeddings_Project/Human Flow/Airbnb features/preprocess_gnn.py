import pandas as pd

# Read the input dataframes from CSV files
dataframe1 = pd.read_csv('Dataset/Raw_Embeddings/Road_Embeddings.csv')  # Replace with your actual file path
dataframe2 = pd.read_csv('Dataset/Final_Embeddings/gcn_llama_onlylist_raw_ver.csv') 

# Ensure dataframe1 and dataframe2 have the expected columns
if 'ADM_NM' not in dataframe1.columns or 'ADM_NM' not in dataframe2.columns or 'Reporting Month' not in dataframe2.columns:
    raise ValueError("Ensure 'ADM_NM' exists in both dataframes and 'Reporting Month' exists in dataframe2")

# Sort dataframe2 to match the order of ADM_NM in dataframe1 for each month
sorted_df2 = pd.concat([
    dataframe2[dataframe2['Reporting Month'] == month]
    .set_index('ADM_NM')
    .loc[dataframe1['ADM_NM']]
    .reset_index()
    for month in dataframe2['Reporting Month'].unique()
])

# Drop the 'ADM_NM' and 'Reporting Month' columns
matrix_df = sorted_df2.drop(columns=['ADM_NM', 'Reporting Month'])

# Convert to a NumPy array for deep learning input
matrix = matrix_df.to_numpy()

# Convert the matrix back to a dataframe without headers for saving or inspection
matrix_without_headers = pd.DataFrame(matrix)

# Save the final matrix as a CSV file without headers
matrix_without_headers.to_csv('quang/time_series_prediction/gnn_embeddings/gcn_llama_onlylist_raw_ver.csv', index=False, header=False)

print("Processing complete. The final matrix is saved as 'gcn_llama_onlylist_raw_ver.csv'.")
