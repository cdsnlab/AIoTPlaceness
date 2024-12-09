import pandas as pd
import re

def format_number(num):
    """
    Format number to maintain scientific notation and precision.
    """
    if abs(num) < 0.0001:  # For very small numbers, use scientific notation
        return f"{num:e}"
    else:
        return f"{num}"

def transform_dataframe(df, embeddings_text, output_path='transformed_data.csv'):
    """
    Transform DataFrame, clean NaN values, and maintain exact number format including scientific notation.
    """
    agg_dict = {
        'Property ID': list,
        'Occupancy Rate': 'mean',
        'Revenue (USD)': 'sum',
        'Number of Reservations': 'sum'
    }

    result_df = df.groupby(['ADM_NM', 'Reporting Month']).agg(agg_dict).reset_index()

    embeddings_list = []
    blocks = embeddings_text.split("--------------------------------------------")

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        embedding = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for prefixed lines starting with "Prompt_{index}:"
            if line.startswith("Prompt_"):
                match = re.search(r"Prompt_\d+:\s*(.*)", line)
                if match:
                    line = match.group(1)  # Extract the part after "Prompt_{index}:"
                else:
                    continue

            # Split numbers by commas or spaces
            parts = [num.strip() for num in line.replace(",", " ").split() if num.strip()]
            try:
                # Convert to float
                numbers = [float(num) for num in parts]
                embedding.extend(numbers)
            except ValueError:
                pass

        if embedding:
            formatted_embedding = [format_number(num) for num in embedding]
            embeddings_list.append(embedding)

    if len(embeddings_list) != len(result_df):
        print(f"Mismatch: {len(embeddings_list)} embeddings for {len(result_df)} rows. Filling missing values.")
        embeddings_list.extend([[] for _ in range(len(result_df) - len(embeddings_list))])

    result_df['LLM Embeddings'] = embeddings_list

    nan_counts = result_df.isna().sum()
    print("\nNaN values found in each column before cleaning:")
    print(nan_counts)

    result_df = result_df.fillna(0)

    result_df_for_csv = result_df.copy()
    result_df_for_csv['Property ID'] = result_df_for_csv['Property ID'].apply(str)
    result_df_for_csv['LLM Embeddings'] = result_df_for_csv['LLM Embeddings'].apply(
        lambda x: str([format_number(num) for num in x]) if x else ''
    )

    result_df_for_csv.to_csv(output_path, index=False)

    print(f"\nTransformed DataFrame dimensions after cleaning:")
    print(f"Number of rows: {result_df.shape[0]}")
    print(f"Number of columns: {result_df.shape[1]}")

    if len(embeddings_list) > 0:
        print("\nVerifying first embedding:")
        print("Length:", len(embeddings_list[0]))
        print("Last 5 numbers:", [format_number(num) for num in embeddings_list[0][-5:]])
        last_numbers = embeddings_list[0][-2:]
        print("\nVerifying last two numbers specifically:")
        print(f"Second to last number (scientific notation): {format_number(last_numbers[0])}")
        print(f"Last number: {format_number(last_numbers[1])}")

    return result_df

# Read the embeddings text
with open('quang/llm_embeddings/embeddings/new/raw_embeddings_new.txt', 'r') as file:
    embeddings_text = file.read()

# Specify the file path
file_path = 'Dataset/preprocess_AirBnB.csv'

# Load a small sample to get a quick overview
df = pd.read_csv(file_path, low_memory=False)

# Transform and save
transformed_df = transform_dataframe(df, embeddings_text, 'Dataset/AirBnB_LLM/gemma2_no_listing_raw.csv')
