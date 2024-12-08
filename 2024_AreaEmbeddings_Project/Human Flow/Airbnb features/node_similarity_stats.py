import pickle
import numpy as np
from datetime import datetime

def calculate_statistics():
    """
    Calculate statistics for similarity scores stored in a Pickle file.
    """
    file_path = 'Dataset/AirBnB_Graph/llama3_no_listing_raw/20170101.gpickle'  # Replace with your file path
    
    # Load data from Pickle file
    print("Loading data from Pickle file...")
    with open(file_path, 'rb') as f:
        similarity_scores = pickle.load(f)
    
    # Check the type of the loaded object
    print(f"Data type of similarity_scores: {type(similarity_scores)}")
    
    # If the loaded data is a dictionary or graph, extract the relevant scores
    if isinstance(similarity_scores, dict):
        similarity_scores = list(similarity_scores.values())
    elif hasattr(similarity_scores, 'edges'):
        similarity_scores = [d['weight'] for _, _, d in similarity_scores.edges(data=True)]

    # Ensure the data is now a flat list or array
    similarity_scores = np.array(similarity_scores)

    # Calculate statistics
    mean = np.mean(similarity_scores)
    std_dev = np.std(similarity_scores)
    median = np.median(similarity_scores)
    min_val = np.min(similarity_scores)
    max_val = np.max(similarity_scores)
    
    # For mode calculation
    unique, counts = np.unique(similarity_scores, return_counts=True)
    mode = unique[np.argmax(counts)]

    # Create results dictionary
    stats = {
        'total_rows': len(similarity_scores),
        'mean': mean,
        'median': median,
        'mode': mode,
        'std_dev': std_dev,
        'min': min_val,
        'max': max_val
    }
    
    return stats

def save_results(stats):
    """
    Save statistics to a file with proper formatting.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'similarity_statistics_{timestamp}.txt'
    
    with open(output_file, 'w') as f:
        f.write("Similarity Score Statistics Report\n")
        f.write("================================\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Statistical Measures:\n")
        f.write("-----------------\n")
        f.write(f"Total Rows Analyzed: {stats['total_rows']:,}\n")
        f.write(f"Mean: {stats['mean']:.6f}\n")
        f.write(f"Median: {stats['median']:.6f}\n")
        f.write(f"Mode: {stats['mode']:.6f}\n")
        f.write(f"Standard Deviation: {stats['std_dev']:.6f}\n")
        f.write(f"Minimum: {stats['min']:.6f}\n")
        f.write(f"Maximum: {stats['max']:.6f}\n\n")
        f.write("================ End of Report ================\n")
    
    return output_file

def main():
    try:
        print("Starting statistical analysis...")
        
        # Calculate statistics
        stats = calculate_statistics()
        
        # Save to file
        output_file = save_results(stats)
        
        # Print results to console
        print("\nResults:")
        print(f"Total Rows: {stats['total_rows']:,}")
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Median: {stats['median']:.6f}")
        print(f"Mode: {stats['mode']:.6f}")
        print(f"Standard Deviation: {stats['std_dev']:.6f}")
        print(f"Minimum: {stats['min']:.6f}")
        print(f"Maximum: {stats['max']:.6f}")
        print(f"\nResults have been saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
