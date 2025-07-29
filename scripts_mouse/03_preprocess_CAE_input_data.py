#!/usr/bin/env python
"""
Preprocess mouse brain spatial transcriptomics data for CAE training.

This script:
1. Loads batch-corrected data for each batch
2. Creates grid representations (32x32) for spatial data
3. Generates sparse matrix representations
4. Saves data loaders for CAE training

Note: This processes each batch separately as they have different spatial arrangements.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path
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
BATCHES = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2']
NORM_SIZE = 32  # Fixed grid size for all mouse batches
BATCH_SIZE = 64
NUM_WORKERS = 8
RANDOM_SEED = 42

# Paths
RESULTS_DIR = "results/"
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)

def load_batch_corrected_data():
    """Load the batch-corrected AnnData object."""
    input_path = os.path.join(PREPROCESSED_DIR, 'mouse_batch_corrected.h5ad')
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Batch-corrected data not found at {input_path}. "
            "Please run batch correction first."
        )
    
    print(f"Loading batch-corrected data from {input_path}")
    adata = sc.read_h5ad(input_path)
    print(f"  Loaded data shape: {adata.shape}")
    
    return adata

def create_batch_specific_adata(adata, batch_id):
    """Create AnnData object for a specific batch."""
    batch_mask = adata.obs['batch'] == batch_id
    adata_batch = adata[batch_mask].copy()
    
    print(f"\n  Batch {batch_id}:")
    print(f"    Spots: {adata_batch.n_obs}")
    print(f"    Genes: {adata_batch.n_vars}")
    
    return adata_batch

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
    
    reshaped_sparse_matrix = coo_matrix((data, (rows, cols)), 
                                       shape=(len(gene_names), norm_size * norm_size))
    return {"sparse_matrix": reshaped_sparse_matrix, "gene_names": gene_names}

def create_cae_input(adata_batch, norm_size, batch_size, num_workers):
    """Create CAE input data loaders for a batch."""
    # Get all genes
    adata_genes = set(adata_batch.var_names)
    train_genes_set = test_genes_set = adata_genes
    
    # Create DataFrame with expression data
    adata_df_observed = pd.DataFrame(
        adata_batch.X, columns=adata_batch.var_names, index=adata_batch.obs_names)
    
    # Add spatial coordinates
    adata_df_observed.insert(0, 'Y', adata_batch.obs['y'])
    adata_df_observed.insert(0, 'X', adata_batch.obs['x'])
    
    # Check data statistics
    len_x_coords = len(adata_df_observed['X'].unique())
    len_y_coords = len(adata_df_observed['Y'].unique())
    print(f"    Number of genes: {len(train_genes_set)}")
    print(f"    Unique coordinates: {len_x_coords} (X), {len_y_coords} (Y)")
    print(f"    X range: {adata_df_observed['X'].min():.0f} to {adata_df_observed['X'].max():.0f}")
    print(f"    Y range: {adata_df_observed['Y'].min():.0f} to {adata_df_observed['Y'].max():.0f}")
    
    # Generate grid with all possible coordinates
    all_combinations = [(x, y) for x in range(norm_size) for y in range(norm_size)]
    new_df = pd.DataFrame(all_combinations, columns=["X", "Y"])
    
    # Merge with observed data, filling missing spots with 0
    adata_df = pd.merge(new_df, adata_df_observed, on=["X", "Y"], how="left").fillna(0)
    adata_df.index = adata_df["X"].astype(str) + "_" + adata_df["Y"].astype(str)
    
    # Create train/test DataFrames (using all genes for both)
    adata_df_train = adata_df[['X', 'Y'] + list(train_genes_set)]
    adata_df_test = adata_df[['X', 'Y'] + list(test_genes_set)]
    
    # Reshape data
    adata_df_train_reshaped = reshape_data(adata_df_train, norm_size)
    adata_df_test_reshaped = reshape_data(adata_df_test, norm_size)
    
    print(f"    Training shape: {adata_df_train_reshaped['sparse_matrix'].shape}")
    print(f"    Array range: {adata_df_train_reshaped['sparse_matrix'].min():.4f} to "
          f"{adata_df_train_reshaped['sparse_matrix'].max():.4f}")
    
    # Create datasets and data loaders
    train_sparse_dataset = SparseDataset(
        adata_df_train_reshaped['sparse_matrix'], 
        adata_df_train_reshaped['gene_names'], 
        norm_size)
    test_sparse_dataset = SparseDataset(
        adata_df_test_reshaped['sparse_matrix'], 
        adata_df_test_reshaped['gene_names'], 
        norm_size)
    
    train_loader = DataLoader(train_sparse_dataset, batch_size, 
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_sparse_dataset, batch_size, 
                           shuffle=False, num_workers=num_workers)
    
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
    print(f"    Saved loader to {os.path.basename(filename)}")

def plot_sample_images(train_loaders, noise_factor=0.0):
    """Plot sample images from each batch."""
    print("\nCreating sample visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for idx, (batch_id, train_loader) in enumerate(train_loaders.items()):
        # Get first batch of data
        first_batch = next(iter(train_loader))
        tensors, gene_names = first_batch
        
        # Original image
        row = idx // 2
        col = (idx % 2) * 2
        ax = axes[row, col]
        
        # Select first image
        img = tensors[0].squeeze().numpy()
        im = ax.imshow(img, cmap='viridis', aspect='auto')
        ax.set_title(f'{batch_id} Original\n{gene_names[0]}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Create noisy version
        if noise_factor > 0:
            noisy_loader = create_noisy_data_loader(train_loader, noise_factor)
            noisy_batch = next(iter(noisy_loader))
            noisy_tensors, _ = noisy_batch
            
            ax = axes[row, col + 1]
            noisy_img = noisy_tensors[0].squeeze().numpy()
            im = ax.imshow(noisy_img, cmap='viridis', aspect='auto')
            ax.set_title(f'{batch_id} Noisy ({noise_factor})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Sample Spatial Gene Expression Images by Batch', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(FIG_DIR, 'mouse_cae_input_samples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved sample images to {output_path}")

def create_noisy_data_loader(original_loader, noise_factor=0.2, num_workers=8):
    """Create noisy version of data loader for visualization."""
    original_dataset = original_loader.dataset
    noisy_dataset = NoisyDataset(original_dataset, noise_factor=noise_factor)
    noisy_loader = DataLoader(
        noisy_dataset, 
        batch_size=original_loader.batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    return noisy_loader

def plot_data_statistics(adata, batches):
    """Plot statistics about the data."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Number of spots per batch
    ax = axes[0, 0]
    spot_counts = adata.obs['batch'].value_counts()[batches]
    spot_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Number of Spots per Batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Number of Spots')
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Gene expression distribution
    ax = axes[0, 1]
    for batch_id in batches:
        batch_data = adata[adata.obs['batch'] == batch_id]
        mean_expression = np.array(batch_data.X.mean(axis=0)).flatten()
        ax.hist(np.log1p(mean_expression), bins=50, alpha=0.5, label=batch_id)
    ax.set_title('Gene Expression Distribution')
    ax.set_xlabel('log(Mean Expression + 1)')
    ax.set_ylabel('Number of Genes')
    ax.legend()
    
    # 3. Spatial coverage
    ax = axes[1, 0]
    for batch_id in batches:
        batch_data = adata.obs[adata.obs['batch'] == batch_id]
        ax.scatter(batch_data['x'], batch_data['y'], 
                  label=batch_id, alpha=0.5, s=10)
    ax.set_title('Spatial Coverage by Batch')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    
    # 4. Sparsity
    ax = axes[1, 1]
    sparsity_data = []
    for batch_id in batches:
        batch_data = adata[adata.obs['batch'] == batch_id]
        sparsity = 1.0 - (np.count_nonzero(batch_data.X) / batch_data.X.size)
        sparsity_data.append(sparsity * 100)
    
    ax.bar(batches, sparsity_data, color='coral')
    ax.set_title('Data Sparsity by Batch')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Sparsity (%)')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Mouse Spatial Transcriptomics Data Statistics', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(FIG_DIR, 'mouse_data_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved statistics plot to {output_path}")

def main():
    """Main preprocessing pipeline for CAE."""
    print("="*80)
    print("Mouse Brain CAE Data Preprocessing")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Load batch-corrected data
    adata = load_batch_corrected_data()
    
    # Plot overall statistics
    plot_data_statistics(adata, BATCHES)
    
    # Process each batch
    print(f"\nProcessing {len(BATCHES)} batches for CAE input...")
    train_loaders = {}
    test_loaders = {}
    
    for batch_id in BATCHES:
        print(f"\nProcessing batch {batch_id}...")
        
        # Get batch-specific data
        adata_batch = create_batch_specific_adata(adata, batch_id)
        
        # Create CAE input
        train_loader, test_loader = create_cae_input(
            adata_batch, NORM_SIZE, BATCH_SIZE, NUM_WORKERS)
        
        train_loaders[batch_id] = train_loader
        test_loaders[batch_id] = test_loader
        
        # Save data loaders
        train_output_file = os.path.join(PREPROCESSED_DIR, 
                                       f"{batch_id}_all_loader.pickle")
        save_data_from_loader(train_loader, train_output_file)
        
        print(f"    Train batches: {len(train_loader)}")
        print(f"    Test batches: {len(test_loader)}")
    
    # Create sample visualizations
    plot_sample_images(train_loaders, noise_factor=0.2)
    
    # Summary
    print("\n" + "="*60)
    print("CAE Preprocessing Summary:")
    print(f"  Processed batches: {len(BATCHES)}")
    print(f"  Grid size: {NORM_SIZE}x{NORM_SIZE}")
    print(f"  Total genes: {adata.n_vars}")
    
    for batch_id in BATCHES:
        print(f"\n  {batch_id}:")
        print(f"    Training batches: {len(train_loaders[batch_id])}")
        print(f"    Samples per batch: {BATCH_SIZE}")
    
    print("\n" + "="*80)
    print("CAE preprocessing complete!")
    print(f"Data saved to: {PREPROCESSED_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print("Next step: Run 04_train_CAE_mouse.py")
    print("="*80)

if __name__ == "__main__":
    main()