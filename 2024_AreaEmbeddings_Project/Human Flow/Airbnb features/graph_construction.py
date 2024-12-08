import os
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from datetime import datetime

# Load data
data = pd.read_csv("quang/similarity_scores/llama3_only_listings.csv")  # Replace with your actual file path

# Create output directory if it doesn't exist
output_dir = "Dataset/AirBnB_Graph/llama3_only_listings"
os.makedirs(output_dir, exist_ok=True)

# Ensure 'month' column is in datetime format
data['month'] = pd.to_datetime(data['month'], errors='coerce')

# Extract unique months
months = data['month'].dt.strftime('%Y%m%d').unique()

# Process each month separately
for month in months:
    # Filter data for the specific month
    month_data = data[data['month'].dt.strftime('%Y%m%d') == month]

    # Extract similarities
    similarities = month_data['similarity_score'].values
    mean_similarity = similarities.mean()              # Mean
    third_quartile = np.quantile(similarities, 0.75)   # 3Q

    # Create a graph
    G = nx.Graph()
    nodes = pd.concat([month_data['group1_district'], month_data['group2_district']]).unique()
    G.add_nodes_from(nodes)

    # Add edges
    for _, row in month_data.iterrows():
        if row['similarity_score'] > third_quartile:
            G.add_edge(row['group1_district'], row['group2_district'], weight=row['similarity_score'])

    # Save graph using pickle
    output_file = os.path.join(output_dir, f"{month}.gpickle")
    with open(output_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph for {month} saved in {output_dir} as Pickle. Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

