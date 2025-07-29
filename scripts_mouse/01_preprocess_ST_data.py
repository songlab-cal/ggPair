#!/usr/bin/env python
"""
Preprocess mouse brain spatial transcriptomics data.

This script:
1. Loads raw count data and metadata from GEO (GSE152506)
2. Filters spots based on metadata
3. Creates AnnData object with spatial coordinates
4. Performs image alignment (flipping and centering)
5. Saves preprocessed data for batch correction

Data source: Alzheimer's disease mouse brain spatial transcriptomics
GEO accession: GSE152506
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
RANDOM_SEED = 42
DATA_DIR = "data/mouse/" 
RESULTS_DIR = "results/"
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")

# Batches to process
BATCHES = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2']

# Batches that need coordinate flipping
BATCHES_TO_FLIP_X = ["B03_C2", "B03_D2", "B05_D2", "B05_E2", "N03_C2", "N03_D2", 
                     "N05_D2", "N05_C2", "N04_D1", "N04_E1"]
BATCHES_TO_FLIP_Y = ["N04_D1", "N04_E1"]

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)

def check_data_exists(data_dir):
    """Check if required data files exist."""
    required_files = [
        'spot_metadata.tsv',
        'GSE152506_raw_counts.txt.gz'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print(f"Please download from GEO (GSE152506) and place in {data_dir}")
        return False
    
    return True

def load_and_filter_data(data_dir):
    """Load raw counts and filter by spot metadata."""
    print("Loading spot metadata...")
    spot_metadata = pd.read_csv(os.path.join(data_dir, 'spot_metadata.tsv'), sep='\t')
    print(f"  Total spots in metadata: {len(spot_metadata)}")
    
    print("\nLoading raw counts (this may take several minutes)...")
    # First, check the column structure
    raw_counts_sample = pd.read_csv(
        os.path.join(data_dir, 'GSE152506_raw_counts.txt.gz'), 
        nrows=5
    )
    print(f"  Column preview: {list(raw_counts_sample.columns[:5])}")
    
    # Load with dask for memory efficiency
    raw_counts = dd.read_csv(
        os.path.join(data_dir, 'GSE152506_raw_counts.txt.gz'), 
        assume_missing=True, 
        sample=500000, 
        blocksize=None
    )
    
    print("Filtering raw counts by spot metadata...")
    filtered_raw_counts = raw_counts[raw_counts['newindex'].isin(spot_metadata['Spot'])]
    
    # Compute and convert to pandas
    print("Computing filtered data (this is the slow step)...")
    filtered_raw_counts = filtered_raw_counts.compute()
    print(f"  Filtered data shape: {filtered_raw_counts.shape}")
    
    # Ensure spot_metadata is in the same order as filtered_raw_counts
    spot_metadata_ordered = spot_metadata.set_index('Spot').loc[filtered_raw_counts['newindex']]
    
    # Set index for counts
    filtered_raw_counts = filtered_raw_counts.set_index('newindex')
    
    return filtered_raw_counts, spot_metadata_ordered

def create_anndata(filtered_raw_counts, spot_metadata_ordered):
    """Create AnnData object with spatial coordinates."""
    print("\nCreating AnnData object...")
    
    # Replace NaN values with 0 and convert to int
    X_data = filtered_raw_counts.values
    X_data = np.nan_to_num(X_data, nan=0).astype(int)
    
    adata = anndata.AnnData(
        X=X_data,
        obs=spot_metadata_ordered,
        var=pd.DataFrame(index=filtered_raw_counts.columns)
    )
    
    print(f"  AnnData shape: {adata.shape}")
    print(f"  Unique batches: {adata.obs['SampleID'].unique()}")
    
    return adata

def align_images(adata, batches_to_flip_x, batches_to_flip_y):
    """Perform image alignment by flipping specific batches."""
    print("\nPerforming image alignment...")
    
    # Rename columns for consistency
    adata.obs.rename(columns={'SampleID': 'batch', 'coord_X': 'x', 'coord_Y': 'y'}, inplace=True)
    
    # Store original coordinates
    adata.obs['x_original'] = adata.obs['x'].copy()
    adata.obs['y_original'] = adata.obs['y'].copy()
    
    # Flip batch-specific images
    for batch_id in batches_to_flip_x:
        if batch_id in adata.obs['batch'].values:
            mask = adata.obs['batch'] == batch_id
            max_x = adata.obs.loc[mask, 'x'].max()
            adata.obs.loc[mask, 'x'] = max_x - adata.obs.loc[mask, 'x']
            print(f"  Flipped X coordinates for batch {batch_id}")
    
    for batch_id in batches_to_flip_y:
        if batch_id in adata.obs['batch'].values:
            mask = adata.obs['batch'] == batch_id
            max_y = adata.obs.loc[mask, 'y'].max()
            adata.obs.loc[mask, 'y'] = max_y - adata.obs.loc[mask, 'y']
            print(f"  Flipped Y coordinates for batch {batch_id}")
    
    # Center each batch to start from (1, 1)
    for batch_id in adata.obs['batch'].unique():
        mask = adata.obs['batch'] == batch_id
        min_x = adata.obs.loc[mask, 'x'].min()
        min_y = adata.obs.loc[mask, 'y'].min()
        adata.obs.loc[mask, 'x'] = adata.obs.loc[mask, 'x'] - min_x + 1
        adata.obs.loc[mask, 'y'] = adata.obs.loc[mask, 'y'] - min_y + 1
    
    # Update spatial coordinates in obsm
    adata.obsm['spatial'] = adata.obs[['x', 'y']].values
    
    return adata

def plot_batch_alignment(adata, output_path):
    """Visualize batch alignment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    # Define colors for tissue types
    tissue_colors = {
        'TH': '#FF6B6B', 'HYP': '#4ECDC4', 'HY': '#3498DB',
        'FB': '#45B7D1', 'HPd': '#96CEB4', 'HPs': '#FFEEAD',
        'CNU': '#D4A5A5', 'CTXsp': '#9B6B6B', 'OLF': '#BBB1D6',
        'MB': '#B19CD9', 'Other': '#808080'
    }
    
    for idx, batch_id in enumerate(BATCHES):
        if idx < len(axes):
            ax = axes[idx]
            batch_data = adata.obs[adata.obs['batch'] == batch_id].copy()
            
            # Mirror X coordinates for visualization
            max_x = batch_data['x'].max()
            batch_data['x'] = max_x - batch_data['x']
            
            # Get tissue types for coloring
            if 'AT' in batch_data.columns:
                for tissue in batch_data['AT'].unique():
                    if tissue in tissue_colors:
                        tissue_data = batch_data[batch_data['AT'] == tissue]
                        ax.scatter(tissue_data['x'], tissue_data['y'], 
                                 c=tissue_colors[tissue], label=tissue, 
                                 s=30, alpha=0.7)
            else:
                ax.scatter(batch_data['x'], batch_data['y'], 
                         c='gray', s=30, alpha=0.7)
            
            ax.set_title(f'{batch_id}')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal')
    
    plt.suptitle('Aligned Spatial Coordinates by Batch', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved alignment plot to {output_path}")

def filter_by_selected_batches(adata, selected_batches):
    """Filter AnnData to include only selected batches."""
    mask = adata.obs['batch'].isin(selected_batches)
    adata_filtered = adata[mask, :].copy()
    print(f"\nFiltered to {len(selected_batches)} batches: {selected_batches}")
    print(f"  Remaining spots: {adata_filtered.n_obs}")
    return adata_filtered

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("Mouse Brain Spatial Transcriptomics Preprocessing")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Check if data exists
    if not check_data_exists(DATA_DIR):
        sys.exit(1)
    
    # Load and filter data
    filtered_counts, spot_metadata = load_and_filter_data(DATA_DIR)
    
    # Create AnnData object
    adata = create_anndata(filtered_counts, spot_metadata)
    
    # Align images
    adata = align_images(adata, BATCHES_TO_FLIP_X, BATCHES_TO_FLIP_Y)
    
    # Filter to selected batches
    adata = filter_by_selected_batches(adata, BATCHES)
    
    # Make gene names unique
    adata.var_names_make_unique()
    
    # Save full aligned data
    output_path = os.path.join(PREPROCESSED_DIR, 'mouse_aligned_raw.h5ad')
    adata.write(output_path)
    print(f"\nSaved aligned data to {output_path}")
    
    # Create visualization
    plot_path = os.path.join(FIG_DIR, 'mouse_batch_alignment.png')
    plot_batch_alignment(adata, plot_path)
    
    # Print summary statistics
    print("\nPreprocessing Summary:")
    print(f"  Total spots: {adata.n_obs}")
    print(f"  Total genes: {adata.n_vars}")
    print(f"  Batches: {list(adata.obs['batch'].unique())}")
    for batch in BATCHES:
        n_spots = sum(adata.obs['batch'] == batch)
        print(f"    {batch}: {n_spots} spots")
    
    print("\n" + "="*80)
    print("Preprocessing complete!")
    print(f"Next step: Run 02_batch_correct_mouse_data.py for batch correction")
    print("="*80)

if __name__ == "__main__":
    main()