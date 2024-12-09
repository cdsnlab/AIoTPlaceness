import pandas as pd
import numpy as np
import ast
import logging
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MonthlyDistrictSimilarityCalculator:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.numerical_cols = ['Revenue (USD)', 'Number of Reservations', 'Occupancy Rate']
        self.scaler = MinMaxScaler()
        self.device = torch.device("cuda:3")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        props = torch.cuda.get_device_properties(self.device)
        logging.info(f"Using GPU: {props.name} with {props.total_memory / 1024**3:.1f} GB")

    def prepare_features(self):
        logging.info("Loading data...")
        df = pd.read_csv(self.input_file, low_memory=False)
        
        df = df.sort_values(['Reporting Month', 'ADM_NM'])
        
        logging.info("Preparing numerical features...")
        numerical_features = self.scaler.fit_transform(df[self.numerical_cols]).astype(np.float32)
        logging.info(f"Numerical features shape: {numerical_features.shape} ({len(self.numerical_cols)} columns)")
        
        logging.info("Preparing embedding features...")
        first_emb = np.array(ast.literal_eval(df['LLM Embeddings'].iloc[0]))
        embedding_dim = len(first_emb)
        logging.info(f"LLM embedding dimension: {embedding_dim}")
        
        embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)
        
        for i in tqdm(range(len(df)), desc="Processing embeddings"):
            embeddings[i] = np.array(ast.literal_eval(df['LLM Embeddings'].iloc[i]), dtype=np.float32)
        
        # Combine and normalize features
        features = np.concatenate([
            numerical_features * np.sqrt(0.5),
            embeddings * np.sqrt(0.5)
        ], axis=1)
        
        logging.info(f"Combined features shape: {features.shape} "
                    f"({numerical_features.shape[1]} numerical + {embedding_dim} embedding = "
                    f"{features.shape[1]} total dimensions)")
        
        features_tensor = torch.from_numpy(features)
        features_norm = torch.norm(features_tensor, dim=1, keepdim=True) + 1e-8 
        features_tensor = features_tensor / features_norm
        
        # Print sample of feature values to verify scaling
        logging.info("\nFeature statistics:")
        logging.info(f"Mean of combined features: {features.mean():.6f}")
        logging.info(f"Std of combined features: {features.std():.6f}")
        logging.info(f"Min of combined features: {features.min():.6f}")
        logging.info(f"Max of combined features: {features.max():.6f}")
        
        return features_tensor, df

    def get_triu_indices(self, n):
        """Get indices for upper triangular part (excluding diagonal)"""
        rows, cols = torch.triu_indices(n, n, offset=1)
        return rows, cols

    def calculate_similarities(self):
        features_tensor, df = self.prepare_features()
        features_gpu = features_tensor.to(self.device)
        
        # Process by month
        total_pairs = 0
        month_groups = df.groupby('Reporting Month')
        # Calculate total pairs across all months
        for _, group in month_groups:
            n_districts = len(group)
            total_pairs += (n_districts * (n_districts - 1)) // 2
            
        logging.info(f"Total pairs to process: {total_pairs:,}")
        
        processed_pairs = 0
        with open(self.output_file, 'w') as f:
            # Write header
            f.write('group1_district,group2_district,month,similarity_score\n')
            
            with tqdm(total=total_pairs, desc="Processing months") as pbar:
                # Process each month
                for month, month_df in month_groups:
                    month_indices = month_df.index
                    if len(month_indices) < 2:
                        continue
                        
                    month_features = features_gpu[month_indices]
                    month_districts = month_df['ADM_NM'].values
                    
                    # Calculate similarities
                    similarities = torch.mm(month_features, month_features.t())
                    
                    # Get upper triangular indices
                    rows, cols = self.get_triu_indices(len(month_indices))
                    triu_similarities = similarities[rows, cols].cpu().numpy()
                    
                    # Write results
                    for idx, (i, j) in enumerate(zip(rows.cpu().numpy(), cols.cpu().numpy())):
                        f.write(f"{month_districts[i]},{month_districts[j]},"
                               f"{month},{triu_similarities[idx]:.8f}\n")
                        processed_pairs += 1
                        pbar.update(1)
                    
                    # Clear GPU cache after each month
                    del similarities
                    torch.cuda.empty_cache()
        
        logging.info(f"Processed {processed_pairs:,} pairs")
        return True

def main():
    input_file = 'Dataset/AirBnB_LLM/llama3_only_listings_fill_na_ver.csv'
    output_file = 'quang/similarity_scores/llama3_only_listings.csv'
    
    start_time = datetime.now()
    
    calculator = MonthlyDistrictSimilarityCalculator(
        input_file=input_file,
        output_file=output_file
    )
    
    try:
        success = calculator.calculate_similarities()
        
        if success:
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logging.info(f"Processing completed in {elapsed_time/3600:.2f} hours")
            
            # Verify output
            df_sample = pd.read_csv(output_file, nrows=1000)
            logging.info(f"Sample statistics (first 1000 rows):")
            logging.info(f"Average similarity: {df_sample['similarity_score'].mean():.8f}")
            logging.info(f"Similarity range: [{df_sample['similarity_score'].min():.8f}, "
                        f"{df_sample['similarity_score'].max():.8f}]")
            
            # Count total pairs in output
            total_pairs = sum(1 for _ in open(output_file)) - 1  # subtract header
            logging.info(f"Total pairs written: {total_pairs:,}")
            
    except Exception as e:
        logging.error(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()