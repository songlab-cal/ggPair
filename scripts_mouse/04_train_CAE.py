#!/usr/bin/env python
"""
Train Convolutional Autoencoders (CAE) for mouse brain spatial transcriptomics.

This script:
1. Trains separate CAE models for each batch
2. Uses denoising autoencoder approach with additive noise
3. Extracts latent embeddings for downstream analysis
4. Creates visualizations of denoising performance

Note: Each batch gets its own model due to potential batch-specific patterns.
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.sparse import coo_matrix
from tqdm import tqdm

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    from utils.datasets import SparseDataset, NoisyDataset
except ImportError:
    print("Error: Cannot import custom Dataset classes.")
    sys.exit(1)

# Configuration
BATCHES = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2']
NORM_SIZE = 32
NOISE_FACTOR = 0.2

# Model parameters
IN_CHANNELS = 1
BASE_FILTERS = 16 
LEARNING_RATE = 3e-4
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 8
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 10
DELAY_EPOCHS = 20

# Paths
RESULTS_DIR = "results/"
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")

SEED = 42

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

class SparseCAE(nn.Module):
    """Sparse Convolutional Autoencoder for spatial transcriptomics."""
    
    def __init__(self, in_channels=1, base_filter=32):
        super(SparseCAE, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_filter, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filter)
        self.enc2 = nn.Conv2d(base_filter, base_filter * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filter * 2)
        self.enc3 = nn.Conv2d(base_filter * 2, base_filter * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filter * 4)
        self.enc4 = nn.Conv2d(base_filter * 4, base_filter * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filter * 8)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(base_filter * 8, base_filter * 4, kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(base_filter * 4)
        self.dec2 = nn.ConvTranspose2d(base_filter * 4, base_filter * 2, kernel_size=2, stride=2)
        self.bn6 = nn.BatchNorm2d(base_filter * 2)
        self.dec3 = nn.ConvTranspose2d(base_filter * 2, base_filter, kernel_size=2, stride=2)
        self.bn7 = nn.BatchNorm2d(base_filter)
        self.dec4 = nn.ConvTranspose2d(base_filter, in_channels, kernel_size=2, stride=2)
        
    def encode(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.enc2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.enc3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.enc4(x)))
        x = self.pool(x)
        return x

    def decode(self, x):
        x = F.relu(self.bn5(self.dec1(x)))
        x = F.relu(self.bn6(self.dec2(x)))
        x = F.relu(self.bn7(self.dec3(x)))
        x = torch.sigmoid(self.dec4(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss

def load_data_to_loader(batch_id, norm_size, preprocessed_dir, batch_size=64, num_workers=8):
    """Load preprocessed data and create train/validation split."""
    file_path = os.path.join(preprocessed_dir, f"{batch_id}_all_loader.pickle")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data not found: {file_path}")
    
    print(f"  Loading data from {file_path}")
    with open(file_path, 'rb') as handle:
        loaded_data = pickle.load(handle)
    
    all_data = np.concatenate(loaded_data['data'], axis=0)
    all_gene_names = [gene for sublist in loaded_data['gene_names'] for gene in sublist]
    total_genes = len(all_gene_names)

    # Recreate sparse matrix
    data, rows, cols = [], [], []
    for gene_idx, gene_data in enumerate(all_data):
        non_zero_indices = np.nonzero(gene_data)
        values = gene_data[non_zero_indices]
        data.extend(values.flatten())
        rows.extend([gene_idx] * len(values))
        cols.extend((non_zero_indices[0] * norm_size + non_zero_indices[1]).flatten())

    reshaped_sparse_matrix = coo_matrix((data, (rows, cols)), 
                                       shape=(total_genes, norm_size * norm_size))

    # Create dataset
    dataset = SparseDataset(reshaped_sparse_matrix, all_gene_names, norm_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Split into train/validation (80/20)
    total_size = len(data_loader.dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(
        data_loader.dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"  Total genes: {total_genes}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return data_loader, train_loader, val_loader

def create_noisy_data_loader(original_loader, noise_factor=0.2, num_workers=8):
    """Create noisy version of data loader."""
    original_dataset = original_loader.dataset
    noisy_dataset = NoisyDataset(original_dataset, noise_factor=noise_factor)
    noisy_loader = DataLoader(
        noisy_dataset, batch_size=original_loader.batch_size, 
        shuffle=False, num_workers=num_workers)
    return noisy_loader

def train_epoch(model, dataloader, noisy_dataloader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(zip(dataloader, noisy_dataloader), total=len(dataloader), desc="Training")
    
    for ((clean_tensors, clean_gene_names), (noisy_tensors, noisy_gene_names)) in pbar:
        assert clean_gene_names == noisy_gene_names, "Gene names mismatch"

        clean_data = clean_tensors.unsqueeze(1).to(device)
        noisy_data = noisy_tensors.unsqueeze(1).to(device)

        optimizer.zero_grad()
        reconstructed = model(noisy_data)
        loss = loss_fn(reconstructed, clean_data)
        
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss detected: {loss.item()}")
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 
                         'mem_gb': torch.cuda.memory_reserved(device)/1e9})
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def validate_model(model, clean_dataloader, noisy_dataloader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for ((clean_tensors, _), (noisy_tensors, _)) in zip(clean_dataloader, noisy_dataloader):
            clean_data = clean_tensors.unsqueeze(1).to(device)
            noisy_data = noisy_tensors.unsqueeze(1).to(device)
            
            outputs = model(noisy_data)
            loss = loss_fn(outputs, clean_data)
            total_loss += loss.item()

    avg_loss = total_loss / len(clean_dataloader)
    return avg_loss

def plot_losses(train_losses, val_losses, batch_id, output_dir):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.title(f'Training History - {batch_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_path = os.path.join(output_dir, f'{batch_id}_training_losses.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_denoising_publication_style(model, clean_dataloader, noise_factor, 
                                        genes_to_plot, batch_id, output_path, 
                                        device, normalize=True):
    """Create publication-quality denoising visualization."""
    noisy_dataloader = create_noisy_data_loader(clean_dataloader, noise_factor)
    model.eval()
    
    # Get data
    clean_images, clean_gene_names = next(iter(clean_dataloader))
    noisy_images, noisy_gene_names = next(iter(noisy_dataloader))
    raw_noisy_images = noisy_images.clone()
    
    # Find indices of requested genes
    gene_indices = []
    found_genes = []
    for gene in genes_to_plot:
        if gene in clean_gene_names:
            gene_indices.append(clean_gene_names.index(gene))
            found_genes.append(gene)
    
    if not gene_indices:
        # If none found, use first 3 genes
        gene_indices = list(range(min(3, len(clean_gene_names))))
        found_genes = [clean_gene_names[i] for i in gene_indices]
        print(f"  Using first {len(found_genes)} genes for visualization")

    # Normalize if requested
    if normalize:
        for i in range(len(clean_images)):
            orig_max = clean_images[i].max()
            if orig_max > 0:
                clean_images[i] = clean_images[i] / orig_max
            
            noisy_max = noisy_images[i].max()
            if noisy_max > 0:
                noisy_images[i] = noisy_images[i] / noisy_max

    # Get denoised images
    with torch.no_grad():
        noisy_subset = raw_noisy_images[gene_indices].unsqueeze(1).to(device)
        denoised_images = model(noisy_subset).cpu().squeeze(1)

    # Create figure
    fig = plt.figure(figsize=(12, 4 * len(gene_indices)))
    gs = fig.add_gridspec(len(gene_indices), 4, width_ratios=[1, 1, 1, 0.05], 
                         hspace=0.3, wspace=0.2)

    vmin, vmax = (0, 1) if normalize else (None, None)

    # Add column titles
    fig.text(0.17, 0.98, 'Original', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.42, 0.98, f'Noisy ({noise_factor})', ha='center', va='top', fontsize=14, fontweight='bold')
    fig.text(0.67, 0.98, 'Denoised', ha='center', va='top', fontsize=14, fontweight='bold')

    for idx, (gene_idx, gene_name) in enumerate(zip(gene_indices, found_genes)):
        # Original
        ax = fig.add_subplot(gs[idx, 0])
        im = ax.imshow(clean_images[gene_idx].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.text(-0.15, 0.5, gene_name, transform=ax.transAxes, 
                rotation=90, va='center', fontsize=12, fontweight='bold')

        # Noisy
        ax = fig.add_subplot(gs[idx, 1])
        ax.imshow(noisy_images[gene_idx].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')

        # Denoised
        ax = fig.add_subplot(gs[idx, 2])
        ax.imshow(denoised_images[idx].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')

        # Colorbar for last row
        if idx == len(gene_indices) - 1:
            cax = fig.add_subplot(gs[:, 3])
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Normalized Expression' if normalize else 'Expression Level', 
                          fontsize=12)

    plt.suptitle(f'Denoising Performance - {batch_id}', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def extract_and_save_embeddings(model, dataloader, batch_id, noise_factor, 
                               embeddings_dir, device):
    """Extract and save flattened latent embeddings."""
    model.eval()
    embeddings_list = []
    gene_names_list = []
    
    with torch.no_grad():
        for tensors, gene_names in tqdm(dataloader, desc="Extracting embeddings"):
            data = tensors.unsqueeze(1).to(device)
            embedding = model.encode(data)
            # Flatten the spatial dimensions
            flattened = embedding.view(embedding.size(0), -1)
            embeddings_list.append(flattened.cpu().numpy())
            gene_names_list.extend(gene_names)

    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings_list)
    
    # Create DataFrame
    embeddings_df = pd.DataFrame(data=all_embeddings, index=gene_names_list)
    
    # Save as CSV
    csv_path = os.path.join(embeddings_dir,
        f"{batch_id}_noise{noise_factor}_gene_embeddings_noisy.csv")
    embeddings_df.to_csv(csv_path)
    
    print(f"  Saved embeddings to {csv_path}")
    print(f"  Embeddings shape: {embeddings_df.shape}")
    
    return embeddings_df

def plot_all_batches_losses(all_histories, output_dir):
    """Create combined loss plots for all batches."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (batch_id, history) in enumerate(all_histories.items()):
        ax = axes[idx]
        ax.plot(history['train_losses'], label='Train', color='blue', linewidth=2)
        ax.plot(history['val_losses'], label='Validation', color='orange', linewidth=2)
        ax.set_title(f'{batch_id}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add final loss values
        final_train = history['train_losses'][-1]
        final_val = history['val_losses'][-1]
        ax.text(0.95, 0.95, f'Final: {final_val:.4f}', 
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('CAE Training History - All Batches', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'mouse_all_batches_training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_cae_for_batch(batch_id, device):
    """Train CAE for a specific batch."""
    print(f"\n{'='*60}")
    print(f"Training CAE for {batch_id}")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data...")
    data_loader, train_loader, val_loader = load_data_to_loader(
        batch_id, NORM_SIZE, PREPROCESSED_DIR, BATCH_SIZE, NUM_WORKERS)
    
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create noisy loaders
    noisy_train_loader = create_noisy_data_loader(train_loader, NOISE_FACTOR)
    noisy_val_loader = create_noisy_data_loader(val_loader, NOISE_FACTOR)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = SparseCAE(IN_CHANNELS, BASE_FILTERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Training setup
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                 patience=LR_SCHEDULER_PATIENCE, verbose=True)
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Training loop
    print(f"\nTraining for up to {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, noisy_train_loader, 
                               loss_fn, optimizer, device)
        # Validate
        val_loss = validate_model(model, val_loader, noisy_val_loader, 
                                loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check (after delay)
        if epoch >= DELAY_EPOCHS:
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 
        f"{batch_id}_noise{NOISE_FACTOR}_sparse_cae_final_model_deeper.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot losses
    plot_losses(train_losses, val_losses, batch_id, FIG_DIR)
    
    # Visualize denoising
    print(f"\nCreating denoising visualization...")
    # Try to use specific genes, otherwise use first 3
    genes_to_plot = ['Tfcp2', 'Csmd1', 'Plec']
    denoising_path = os.path.join(FIG_DIR, f'{batch_id}_denoising_examples.png')
    visualize_denoising_publication_style(
        model, val_loader, NOISE_FACTOR, genes_to_plot, 
        batch_id, denoising_path, device, normalize=True)
    
    # Extract embeddings from full dataset with noise
    print(f"\nExtracting embeddings...")
    noisy_data_loader = create_noisy_data_loader(data_loader, NOISE_FACTOR)
    embeddings_df = extract_and_save_embeddings(
        model, noisy_data_loader, batch_id, NOISE_FACTOR, EMBEDDINGS_DIR, device)
    
    # Create history dictionary
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'embedding_shape': embeddings_df.shape
    }
    
    return model, history

def main():
    """Main training pipeline."""
    print("="*80)
    print("Mouse Brain CAE Training Pipeline")
    print("="*80)
    
    set_seed(SEED)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available, training will be slow!")
    
    # Create output directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Train models for each batch
    all_histories = {}
    
    for batch_id in BATCHES:
        try:
            model, history = train_cae_for_batch(batch_id, device)
            all_histories[batch_id] = history
            
            # Save history
            history_path = os.path.join(MODELS_DIR, f"{batch_id}_training_history.pickle")
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
            
            # Clear GPU cache between batches
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError training {batch_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined visualizations
    if all_histories:
        plot_all_batches_losses(all_histories, FIG_DIR)
    
    # Save all histories
    all_histories_path = os.path.join(MODELS_DIR, "all_batches_training_history.pickle")
    with open(all_histories_path, 'wb') as f:
        pickle.dump(all_histories, f)
    
    print("\n" + "="*80)
    print("CAE training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Embeddings saved to: {EMBEDDINGS_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print("\nNext steps:")
    print("1. Run SNN training for 5-fold CV")
    print("2. Run LOOCV analysis")
    print("="*80)

if __name__ == "__main__":
    main()