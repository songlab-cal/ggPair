#!/usr/bin/env python
"""
Perform batch correction on mouse brain spatial transcriptomics data using GraphST.

This script:
1. Loads aligned data from preprocessing
2. Applies GraphST batch correction
3. Performs clustering (mclust requires R, or use leiden/louvain)
4. Saves batch-corrected data and visualizations

Note: For mclust clustering, R must be installed and R_HOME must be set.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from GraphST import GraphST

# Configuration
RANDOM_SEED = 42
RESULTS_DIR = "results/"
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")

# GraphST parameters
N_CLUSTERS = 10
CLUSTERING_METHOD = 'mclust'  # Options: 'mclust' (requires R), 'leiden', 'louvain'

# R configuration (only needed for mclust)
# Adjust this path to the R installation
R_HOME = '/usr/lib/R'  

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def setup_environment():
    """Set up environment for GraphST."""
    # Set R_HOME if using mclust
    if CLUSTERING_METHOD == 'mclust':
        if not os.path.exists(R_HOME):
            print(f"Warning: R_HOME path '{R_HOME}' does not exist!")
            print("Please install R or switch to 'leiden' or 'louvain' clustering.")
            print("You can install R using: conda install -c conda-forge r-base")
            return False
        os.environ['R_HOME'] = R_HOME
    
    # Check for GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    return True

def load_aligned_data(preprocessed_dir):
    """Load aligned data from preprocessing step."""
    input_path = os.path.join(preprocessed_dir, 'mouse_aligned_raw.h5ad')
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Aligned data not found at {input_path}. "
            "Please run 01_preprocess_mouse_data.py first.")
    
    print(f"Loading aligned data from {input_path}")
    adata = sc.read_h5ad(input_path)
    print(f"  Loaded data shape: {adata.shape}")
    
    return adata

def run_graphst_batch_correction(adata):
    """Run GraphST for batch correction."""
    print("\nRunning GraphST batch correction...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize GraphST model
    model = GraphST.GraphST(adata, device=device)
    
    # Train model
    print("  Training GraphST model...")
    adata = model.train()
    
    print("  Batch correction complete!")
    print(f"  Output data type: {adata.X.dtype}")
    
    return adata

def perform_clustering(adata, n_clusters, method='mclust'):
    """Perform clustering on batch-corrected data."""
    print(f"\nPerforming {method} clustering with {n_clusters} clusters...")
    
    from GraphST.utils import clustering
    
    if method == 'mclust':
        try:
            clustering(adata, n_clusters, method=method)
        except Exception as e:
            print(f"Error with mclust clustering: {e}")
            print("Falling back to leiden clustering...")
            method = 'leiden'
    
    if method in ['leiden', 'louvain']:
        clustering(adata, n_clusters, method=method, start=0.1, end=2.0, increment=0.01)
    
    print(f"  Clustering complete! Found clusters: {adata.obs['domain'].unique()}")
    
    return adata

def create_batch_correction_plots(adata, output_dir):
    """Create visualization plots for batch correction results."""
    print("\nCreating visualization plots...")
    
    fig, ax_list = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. UMAP before batch correction (using PCA on raw data)
    print("  Computing UMAP on uncorrected data...")
    adata_copy = adata.copy()
    sc.pp.normalize_total(adata_copy)
    sc.pp.log1p(adata_copy)
    sc.pp.pca(adata_copy)
    sc.pp.neighbors(adata_copy, use_rep='X_pca', n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata_copy)
    
    sc.pl.umap(adata_copy, color='batch', title='Before Batch Correction',
               ax=ax_list[0], show=False, legend_loc='on data')
    
    # 2. UMAP after batch correction
    print("  Computing UMAP on corrected data...")
    sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10)
    sc.tl.umap(adata)
    
    sc.pl.umap(adata, color='batch', title='After Batch Correction',
               ax=ax_list[1], show=False, legend_loc='on data')
    
    # 3. UMAP colored by clusters
    sc.pl.umap(adata, color='domain', title='Clusters',
               ax=ax_list[2], show=False)
    
    plt.suptitle('GraphST Batch Correction Results', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mouse_batch_correction_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")

def plot_spatial_domains(adata, output_dir):
    """Plot spatial domains for each batch."""
    print("\nPlotting spatial domains...")
    
    n_batches = len(adata.obs['batch'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, batch_id in enumerate(adata.obs['batch'].unique()):
        if idx < len(axes):
            ax = axes[idx]
            batch_data = adata[adata.obs['batch'] == batch_id]
            
            # Use spatial coordinates
            spatial_coords = batch_data.obsm['spatial']
            domains = batch_data.obs['domain']
            
            scatter = ax.scatter(spatial_coords[:, 0], spatial_coords[:, 1],
                               c=domains.astype(int), cmap='tab20',
                               s=30, alpha=0.7)
            
            ax.set_title(f'{batch_id} - Spatial Domains')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')
    
    plt.suptitle('Spatial Domains by Batch', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mouse_spatial_domains.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")

def main():
    """Main batch correction pipeline."""
    print("="*80)
    print("Mouse Brain Spatial Transcriptomics Batch Correction")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    # Setup environment
    if not setup_environment():
        print("Environment setup failed. Please check R installation if using mclust.")
        if CLUSTERING_METHOD == 'mclust':
            print("Tip: You can change CLUSTERING_METHOD to 'leiden' or 'louvain' to avoid R dependency.")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(FIG_DIR, exist_ok=True)
    
    try:
        # Load aligned data
        adata = load_aligned_data(PREPROCESSED_DIR)
        
        # Run batch correction
        adata = run_graphst_batch_correction(adata)
        
        # Perform clustering
        adata = perform_clustering(adata, N_CLUSTERS, CLUSTERING_METHOD)
        
        # Create visualizations
        create_batch_correction_plots(adata, FIG_DIR)
        plot_spatial_domains(adata, FIG_DIR)
        
        # Save batch-corrected data
        output_path = os.path.join(PREPROCESSED_DIR, 'mouse_batch_corrected.h5ad')
        adata.write(output_path)
        print(f"\nSaved batch-corrected data to {output_path}")
        
        # Print summary
        print("\nBatch Correction Summary:")
        print(f"  Total spots: {adata.n_obs}")
        print(f"  Total genes: {adata.n_vars}")
        print(f"  Number of clusters: {len(adata.obs['domain'].unique())}")
        print(f"  Embedding shape: {adata.obsm['emb_pca'].shape}")
        
    except Exception as e:
        print(f"\nError during batch correction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Batch correction complete!")
    print(f"Next step: Run CAE training script")
    print("="*80)

if __name__ == "__main__":
    main()