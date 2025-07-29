#!/usr/bin/env python
"""
Train and evaluate Convolutional Autoencoder (CAE) for spatial transcriptomics denoising.

The CAE is trained to denoise spatial gene expression patterns by learning
to reconstruct clean images from noisy inputs. After training, it extracts
latent embeddings for downstream SNN training.
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.sparse import coo_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

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
DATASETS = [7, 23, 24]
NORM_SIZES = {7: 64, 23: 128, 24: 128}
NOISE_FACTORS = {7: 1.0, 23: 1.0, 24: 1.0}

# Model parameters
IN_CHANNELS = 1
BASE_FILTERS = 8
LEARNING_RATE = 3e-4
EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 8
EARLY_STOPPING_PATIENCE = 20
LR_SCHEDULER_PATIENCE = 10

# Random seed
SEED = 42

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
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
        x = torch.sigmoid(self.dec4(x))  # Sigmoid activation for final layer
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.val_loss_min = val_loss

def load_data_to_loader(dataset_idx, norm_size, preprocessed_dir, batch_size=32, num_workers=8):
    """Load preprocessed data and create train/validation split."""
    file_path = os.path.join(preprocessed_dir, f"Dataset{dataset_idx}_all_loader_unnorm01.pickle")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data not found: {file_path}")
    
    print(f"  Loading preprocessed data from {file_path}")
    with open(file_path, 'rb') as handle:
        loaded_data = pickle.load(handle)
    
    all_data = np.concatenate(loaded_data['data'], axis=0)
    all_gene_names = [gene for sublist in loaded_data['gene_names'] for gene in sublist]
    total_genes = len(all_gene_names)

    # Recreate the sparse matrix from the loaded data
    data, rows, cols = [], [], []
    for gene_idx, gene_data in enumerate(all_data):
        non_zero_indices = np.nonzero(gene_data)
        values = gene_data[non_zero_indices]
        data.extend(values.flatten())
        rows.extend([gene_idx] * len(values))
        cols.extend((non_zero_indices[0] * norm_size + non_zero_indices[1]).flatten())

    reshaped_sparse_matrix = coo_matrix((data, (rows, cols)), shape=(total_genes, norm_size * norm_size))

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
    
    return data_loader, train_loader, val_loader

def create_noisy_data_loader(original_loader, noise_factor=1.0, num_workers=8):
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
    
    pbar = tqdm(zip(dataloader, noisy_dataloader), total=len(dataloader))
    
    for ((clean_tensors, clean_gene_names), (noisy_tensors, noisy_gene_names)) in pbar:
        assert clean_gene_names == noisy_gene_names, "Gene names mismatch between clean and noisy data"

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
        pbar.set_description(f"Loss: {loss.item():.4f}, Mem: {torch.cuda.memory_reserved(device)/1e9:.2f}GB")
    
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

def plot_losses(train_losses, val_losses, output_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_denoising(model, clean_dataloader, noise_factor, gene_names_to_plot, 
                       output_path, device, normalize_or_not=True):
    """Visualize original, noisy, and denoised images for specific genes."""
    noisy_dataloader = create_noisy_data_loader(clean_dataloader, noise_factor)
    model.eval()
    
    clean_images, clean_gene_names = next(iter(clean_dataloader))
    noisy_images, noisy_gene_names = next(iter(noisy_dataloader))
    raw_noisy_images = noisy_images.clone()
    
    # Find indices of requested genes
    gene_indices = []
    for gene in gene_names_to_plot:
        if gene in clean_gene_names:
            gene_indices.append(clean_gene_names.index(gene))
    
    if not gene_indices:
        print(f"Warning: None of the requested genes found in batch")
        return

    # Optionally normalize
    if normalize_or_not:
        for i in range(len(clean_images)):
            orig_max = clean_images[i].max() * 0.8
            if orig_max > 0:
                clean_images[i] /= orig_max
            
            noisy_max = noisy_images[i].max() * 0.8
            if noisy_max > 0:
                noisy_images[i] /= noisy_max

    with torch.no_grad():
        noisy_images_subset = raw_noisy_images[gene_indices].unsqueeze(1).to(device)
        denoised_images = model(noisy_images_subset).cpu().squeeze(1)

    # Set up the figure
    fig = plt.figure(figsize=(12, 4 * len(gene_indices)))
    gs = fig.add_gridspec(len(gene_indices), 4, width_ratios=[1, 1, 1, 0.05])

    vmin, vmax = (0, 1) if normalize_or_not else (None, None)

    for idx, (i, gene_name) in enumerate(zip(gene_indices, gene_names_to_plot[:len(gene_indices)])):
        # Original
        ax = fig.add_subplot(gs[idx, 0])
        im = ax.imshow(clean_images[i].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title('Original' if idx == 0 else '')
        ax.axis('off')
        ax.text(-0.2, 0.5, gene_name, transform=ax.transAxes, 
                rotation=90, va='center', fontsize=12)

        # Noisy
        ax = fig.add_subplot(gs[idx, 1])
        ax.imshow(noisy_images[i].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title('Noisy' if idx == 0 else '')
        ax.axis('off')

        # Denoised
        ax = fig.add_subplot(gs[idx, 2])
        ax.imshow(denoised_images[idx].numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title('Denoised' if idx == 0 else '')
        ax.axis('off')

        # Add colorbar for last row
        if idx == len(gene_indices) - 1:
            cax = fig.add_subplot(gs[:, 3])
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Normalized Expression Level', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved denoising visualization to {output_path}")

def extract_latent_embeddings(model, dataloader, device):
    """Extract latent embeddings from the encoder."""
    model.eval()
    embeddings = []
    gene_names_list = []
    
    with torch.no_grad():
        for tensors, gene_names in tqdm(dataloader, desc="Extracting embeddings"):
            data = tensors.unsqueeze(1).to(device)
            embedding = model.encode(data)
            embeddings.append(embedding.cpu().numpy())
            gene_names_list.extend(gene_names)
    
    return np.concatenate(embeddings), gene_names_list

def extract_flattened_latent_embeddings_to_df(model, dataloader, dataset_idx, noise_factor, 
                                             embeddings_dir, device):
    """Extract flattened latent embeddings and save as DataFrame."""
    model.eval()
    embeddings_list = []
    gene_names_list = []
    
    with torch.no_grad():
        for tensors, gene_names in tqdm(dataloader, desc="Extracting embeddings"):
            data = tensors.unsqueeze(1).to(device)
            embedding = model.encode(data)
            flattened = embedding.view(embedding.size(0), -1)
            embeddings_list.append(flattened.cpu().numpy())
            gene_names_list.extend(gene_names)

    all_embeddings = np.concatenate(embeddings_list)
    embeddings_df = pd.DataFrame(data=all_embeddings, index=gene_names_list)
    
    csv_file_path = os.path.join(embeddings_dir,
        f"dataset{dataset_idx}_noise{noise_factor}_gene_embeddings_noisy.csv")
    embeddings_df.to_csv(csv_file_path)
    print(f"  Saved embeddings to {csv_file_path}")
    print(f"  Embeddings shape: {embeddings_df.shape}")
    
    return embeddings_df

def train_cae_for_dataset(dataset_idx, device):
    """Train CAE for a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Training CAE for Dataset {dataset_idx}")
    print(f"{'='*60}")
    
    norm_size = NORM_SIZES[dataset_idx]
    noise_factor = NOISE_FACTORS[dataset_idx]
    
    # Setup paths
    project_root = parent_dir
    preprocessed_dir = os.path.join(project_root, 'results', 'preprocessed')
    models_dir = os.path.join(project_root, 'results', 'models')
    fig_dir = os.path.join(project_root, 'results', 'fig')
    embeddings_dir = os.path.join(project_root, 'results', 'embeddings')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data (norm_size={norm_size}, noise_factor={noise_factor})...")
    data_loader, train_loader, val_loader = load_data_to_loader(
        dataset_idx, norm_size, preprocessed_dir, BATCH_SIZE, NUM_WORKERS)
    
    print(f"  Total samples: {len(data_loader.dataset)}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create noisy loaders
    noisy_train_loader = create_noisy_data_loader(train_loader, noise_factor)
    noisy_val_loader = create_noisy_data_loader(val_loader, noise_factor)
    
    # Initialize model
    print(f"\nInitializing model...")
    model = SparseCAE(IN_CHANNELS, BASE_FILTERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Schedulers
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=LR_SCHEDULER_PATIENCE, verbose=True)
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, noisy_train_loader, loss_fn, optimizer, device)
        val_loss = validate_model(model, val_loader, noisy_val_loader, loss_fn, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Check early stopping
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
    
    # Save model
    model_path = os.path.join(models_dir, 
        f"100sigmoid_dataset{dataset_idx}_noise{noise_factor}_sparse_cae_final_model_deeper.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot losses
    loss_plot_path = os.path.join(fig_dir, f"dataset{dataset_idx}_training_losses.png")
    plot_losses(train_losses, val_losses, loss_plot_path)
    print(f"Loss plot saved to {loss_plot_path}")
    
    # ==== EVALUATION SECTION ====
    print(f"\n{'='*40}")
    print("Model Evaluation")
    print(f"{'='*40}")
    
    # # Visualize denoising for select genes
    # if dataset_idx == 7:
    #     genes_to_plot = ['LAMA1', 'TOR1B', 'NDUFB5']
    # elif dataset_idx == 23:
    #     genes_to_plot = ['NANS', 'OGT', 'MMP3']
    # else:  # dataset 24
    #     genes_to_plot = ['NANS', 'OGT', 'MMP3']
    val_first_imgs, val_first_names = next(iter(val_loader))
    genes_to_plot = list(val_first_names)[:3]
    print(f"[viz] auto-selected genes from first val batch: {genes_to_plot}")
        
    denoising_plot_path = os.path.join(fig_dir, f"dataset{dataset_idx}_denoising_examples.png")
    visualize_denoising(model, val_loader, noise_factor, genes_to_plot, 
                       denoising_plot_path, device, normalize_or_not=True)
    
    # Extract latent embeddings from full dataset
    print(f"\nExtracting latent embeddings...")
    noisy_data_loader = create_noisy_data_loader(data_loader, noise_factor)
    embeddings_df = extract_flattened_latent_embeddings_to_df(
        model, noisy_data_loader, dataset_idx, noise_factor, embeddings_dir, device)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs_trained': len(train_losses),
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'embedding_shape': embeddings_df.shape
    }
    
    history_path = os.path.join(models_dir, f"dataset{dataset_idx}_training_history.pickle")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    return model, history

def main():
    """Main training pipeline."""
    print("="*80)
    print("ggPair CAE Training and Evaluation Pipeline")
    print("="*80)
    
    set_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available, training will be slow!")
    
    # Train models for each dataset
    all_histories = {}
    
    for dataset_idx in DATASETS:
        try:
            model, history = train_cae_for_dataset(dataset_idx, device)
            all_histories[dataset_idx] = history
            
            # Clear GPU cache between datasets
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\nError training dataset {dataset_idx}: {e}")
            continue
    
    print("\n" + "="*80)
    print("CAE training and evaluation complete!")
    print(f"Models saved to: results/models/")
    print(f"Plots saved to: results/fig/")
    print(f"Embeddings saved to: results/embeddings/")
    print("="*80)

if __name__ == "__main__":
    main()