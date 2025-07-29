#!/usr/bin/env python
"""
Preprocess spatial transcriptomics data for ggPair pipeline.

Data download instructions:
1. Install gdown: pip install gdown && export PATH="$HOME/.local/bin:$PATH"
2. Download data: gdown 1BiXROO5eetEInRwMbiA51Aw1eKo2v2Cx -O data/spatial_benchmark.zip
3. Extract only Dataset7, Dataset23, and Dataset24: cd data && unzip spatial_benchmark.zip "DataUpload/Dataset7/*" "DataUpload/Dataset23/*" "DataUpload/Dataset24/*"
   This will create data/DataUpload/ directory with all datasets

Dataset 7 corresponds to Human osteosarcoma data (BC22).
Dataset 23 corresponds to Human breast cancer data (CID4465).
Dataset 24 corresponds to Human breast cancer data (CID44971).
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path to import custom modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    from utils.datasets import SparseDataset, NoisyDataset
except ImportError:
    print("Error: Cannot import custom Dataset classes.")
    print("Please ensure utils/datasets.py exists with SparseDataset and NoisyDataset classes")
    sys.exit(1)

# Configuration
DATASETS = [7, 23, 24]
NORM_SIZES = {7: 64, 23: 128, 24: 128}
BATCH_SIZE = 32
NUM_WORKERS = 6
RANDOM_SEED = 42

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)

def check_data_exists(data_dir):
    """Check if data directory exists and has required structure."""
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("\nPlease download the data first:")
        print("1. pip install gdown")
        print("2. gdown 1BiXROO5eetEInRwMbiA51Aw1eKo2v2Cx -O data/spatial_benchmark.zip")
        print("3. cd data && unzip spatial_benchmark.zip")
        return False
    
    # Check for DataUpload subdirectory
    dataupload_dir = os.path.join(data_dir, 'DataUpload')
    if not os.path.exists(dataupload_dir):
        print(f"Error: DataUpload directory not found in '{data_dir}'!")
        print("Please ensure you've extracted spatial_benchmark.zip correctly.")
        return False
    
    # Check for required datasets
    missing_datasets = []
    for dataset_id in DATASETS:
        dataset_dir = os.path.join(dataupload_dir, f'Dataset{dataset_id}')
        if not os.path.exists(dataset_dir):
            missing_datasets.append(dataset_id)
    
    if missing_datasets:
        print(f"Error: Missing datasets: {missing_datasets}")
        return False
    
    return True

def normalize_to_pixels(series, norm_size):
    """Normalize series to 0-(norm_size-1) range."""
    min_val = series.min()
    max_val = series.max()
    series_normalized = (series - min_val) / (max_val - min_val)
    series_scaled = np.round(series_normalized * (norm_size - 1)).astype(int)
    return series_scaled

def process_dataset(dataset_number, norm_size, data_dir, fig_dir):
    """Process a single dataset and create visualizations."""
    print(f"\nProcessing Dataset {dataset_number} (norm_size={norm_size})...")
    
    # Construct file paths
    dataset_dir = os.path.join(data_dir, 'DataUpload', f'Dataset{dataset_number}')
    
    # Load data files
    try:
        spatial_data = pd.read_csv(os.path.join(dataset_dir, 'Spatial_count.txt'), sep="\t")
        location_data = pd.read_csv(os.path.join(dataset_dir, 'Locations.txt'), sep="\t")
        gene_pairs = pd.read_csv(os.path.join(dataset_dir, f'filtered_pairs_{dataset_number}.csv'), index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading data files for Dataset {dataset_number}: {e}")
        return None, None
    
    # Create AnnData object
    adata = anndata.AnnData(
        X=spatial_data, obs=location_data,
        var=pd.DataFrame(index=spatial_data.columns))
    
    # Log normalize the data
    print("  Normalizing gene expression data...")
    sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    
    # Get unique L-R genes
    unique_LR_genes = pd.unique(gene_pairs[['ligand_name', 'receptor_name']].values.ravel('K'))
    
    # Store original coordinate ranges
    min_x_unnorm, max_x_unnorm = adata.obs.x.min(), adata.obs.x.max()
    min_y_unnorm, max_y_unnorm = adata.obs.y.min(), adata.obs.y.max()

    # Apply normalization to x and y coordinates
    adata.obs.rename(columns={'x': 'x_unnorm', 'y': 'y_unnorm'}, inplace=True)
    adata.obs['x'] = normalize_to_pixels(adata.obs['x_unnorm'], norm_size)
    adata.obs['y'] = normalize_to_pixels(adata.obs['y_unnorm'], norm_size)
    adata.obs['spot'] = adata.obs['x'].astype(str) + "_" + adata.obs['y'].astype(str)
    adata.obs.index = adata.obs['spot']

    # Create visualization
    print("  Creating coordinate visualization...")
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot normalized data
    min_x_norm = round(adata.obs['x'].min(), 2)
    max_x_norm = round(adata.obs['x'].max(), 2)
    min_y_norm = round(adata.obs['y'].min(), 2)
    max_y_norm = round(adata.obs['y'].max(), 2)
    axs[0].scatter(adata.obs['x'], adata.obs['y'], alpha=0.6)
    axs[0].set_title(f"Normalized\nX range: {min_x_norm} to {max_x_norm}\nY range: {min_y_norm} to {max_y_norm}")
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')

    # Plot unnormalized data
    axs[1].scatter(adata.obs['x_unnorm'], adata.obs['y_unnorm'], alpha=0.6)
    axs[1].set_title(f"Unnormalized\nX range: {round(min_x_unnorm, 2)} to {round(max_x_unnorm, 2)}\nY range: {round(min_y_unnorm, 2)} to {round(max_y_unnorm, 2)}")
    axs[1].set_xlabel('X Coordinate')
    axs[1].set_ylabel('Y Coordinate')
    
    plt.suptitle(f'Dataset {dataset_number} - Spatial Coordinate Normalization')
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, f'dataset{dataset_number}_coordinates.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved coordinate plot to {fig_path}")

    return adata, unique_LR_genes

def reshape_data(df, norm_size):
    """Reshape data for CAE input."""
    df_without_coordinates = df.drop(['X', 'Y'], axis=1)
    gene_names = df_without_coordinates.columns.tolist()
    
    data, rows, cols = [], [], []
    for gene_idx, gene in enumerate(gene_names):
        series = df_without_coordinates[gene]
        non_zero_indices = np.nonzero(series.to_numpy())[0]
        for idx in non_zero_indices:
            data.append(series.iloc[idx])
            rows.append(gene_idx)
            cols.append(idx)
    
    reshaped_sparse_matrix = coo_matrix((data, (rows, cols)), shape=(len(gene_names), norm_size * norm_size))
    return {"sparse_matrix": reshaped_sparse_matrix, "gene_names": gene_names}

def create_cae_input(adata, unique_LR_genes, norm_size, batch_size=32, num_workers=6):
    """Create CAE input data loaders."""
    adata_genes = set(adata.var_names)
    train_genes_set = test_genes_set = adata_genes  # Use all genes for both train and test

    # Create DataFrame with spatial coordinates
    adata_df_observed = pd.DataFrame(
        adata.X, columns=adata.var_names, index=adata.obs_names)
    adata_df_observed.insert(0, 'Y', adata.obs['y'])
    adata_df_observed.insert(0, 'X', adata.obs['x'])

    # Print data statistics
    len_x_coords = len(adata_df_observed.iloc[:, 0].unique())
    len_y_coords = len(adata_df_observed.iloc[:, 1].unique())
    print(f"  Number of genes: {len(train_genes_set)} (training), {len(test_genes_set)} (test)") 
    print(f"  Number of unique coordinates: {len_x_coords} (X), {len_y_coords} (Y)")
    print(f"  X range: {adata_df_observed.iloc[:, 0].min()} to {adata_df_observed.iloc[:, 0].max()}")
    print(f"  Y range: {adata_df_observed.iloc[:, 1].min()} to {adata_df_observed.iloc[:, 1].max()}")

    # Generate all possible X and Y combinations for a grid
    all_combinations = [(x, y) for x in range(norm_size) for y in range(norm_size)]
    new_df = pd.DataFrame(all_combinations, columns=["X", "Y"])
    # Fill missing values with zero
    adata_df = pd.merge(new_df, adata_df_observed, on=["X", "Y"], how="left").fillna(0)
    adata_df.index = adata_df["X"].astype(str) + "_" + adata_df["Y"].astype(str)

    adata_df_train = adata_df[['X', 'Y'] + list(train_genes_set)]
    adata_df_test = adata_df[['X', 'Y'] + list(test_genes_set)]

    # Reshape the training and testing data
    adata_df_train_reshaped = reshape_data(adata_df_train, norm_size)
    adata_df_test_reshaped = reshape_data(adata_df_test, norm_size)

    print(f"  Training shape: {adata_df_train_reshaped['sparse_matrix'].shape}, "
          f"Testing shape: {adata_df_test_reshaped['sparse_matrix'].shape}")
    print(f"  Array range: {adata_df_train_reshaped['sparse_matrix'].min():.4f} to "
          f"{adata_df_train_reshaped['sparse_matrix'].max():.4f}")
    
    # Creating Datasets and DataLoaders with gene names
    train_sparse_dataset = SparseDataset(
        adata_df_train_reshaped['sparse_matrix'], 
        adata_df_train_reshaped['gene_names'], 
        norm_size)
    test_sparse_dataset = SparseDataset(
        adata_df_test_reshaped['sparse_matrix'], 
        adata_df_test_reshaped['gene_names'], 
        norm_size)

    train_loader = DataLoader(train_sparse_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_sparse_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def save_data_from_loader(loader, filename):
    """Save DataLoader contents to pickle file."""
    all_data = []
    all_gene_names = []
    for data, gene_names in loader:
        all_data.append(data.numpy())
        all_gene_names.append(gene_names)
    
    with open(filename, 'wb') as handle:
        pickle.dump({"data": all_data, "gene_names": all_gene_names}, handle, protocol=4)
    print(f"  Saved loader data to {filename}")

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("ggPair Data Preprocessing Pipeline")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'results')
    fig_dir = os.path.join(results_dir, 'fig')
    preprocessed_dir = os.path.join(results_dir, 'preprocessed')
    
    # Create output directories
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    
    # Check if data exists
    if not check_data_exists(data_dir):
        sys.exit(1)
    
    # Process each dataset
    for dataset_idx in DATASETS:
        norm_size = NORM_SIZES[dataset_idx]
        
        # Process dataset
        adata, unique_LR_genes = process_dataset(dataset_idx, norm_size, data_dir, fig_dir)
        if adata is None:
            continue
        
        # Create CAE input
        print(f"\n  Creating CAE input for Dataset {dataset_idx}...")
        train_loader, test_loader = create_cae_input(
            adata, unique_LR_genes, norm_size, 
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        # Save processed data
        output_file = os.path.join(preprocessed_dir, f"Dataset{dataset_idx}_all_loader_unnorm01.pickle")
        save_data_from_loader(train_loader, output_file)
        
        print(f"\n  Dataset {dataset_idx} processing complete!")
        print(f"  Number of training batches: {len(train_loader)}")
        print(f"  Number of test batches: {len(test_loader)}")
    
    print("\n" + "="*80)
    print("Preprocessing complete!")
    print(f"Figures saved to: {fig_dir}")
    print(f"Preprocessed data saved to: {preprocessed_dir}")
    print("="*80)

if __name__ == "__main__":
    main()