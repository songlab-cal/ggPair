#!/usr/bin/env python
"""
Train Siamese Neural Network (SNN) using Leave-One-Out Cross-Validation (LOOCV).

This script:
1. Performs leave-one-receptor-out cross-validation
2. Can focus on GPCR or non-GPCR receptors
3. Handles FZD gene exclusion options
4. Processes each batch separately

Note: Mouse genes use lowercase (e.g., 'Fzd' not 'FZD')
"""

import argparse
def parse_arguments():
    parser = argparse.ArgumentParser(description="ggPair LOOCV script")
    parser.add_argument("--dataset_index", type=str, default='B02_D1', help="Dataset index (default: B02_D1)")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device to use (default: '0')")
    parser.add_argument("--neg_pos_ratio", type=int, default=10, help="Negative to positive ratio (default: 10)")
    parser.add_argument("--noise_factor", type=float, default=1.0, help="Noise factor (default: 1.0)")
    parser.add_argument("--gpcr_only", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use only GPCR data (default: True)")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight Decay (default: 1e-3)")
    parser.add_argument("--fzd_approach", type=str, choices=['include', 'exclude', 'exclude_val'], default='include', 
                        help="Approach for handling FZD genes: include, exclude, or exclude_val (default: include)")
    return parser.parse_args()

import os
args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys
import gc
import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import hypergeom
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Sampler, Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
BATCHES = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2']
LATENT_DIM = 256
NEG_POS_RATIO = 10
NOISE_FACTOR = 0.2
BATCH_SIZE = 99
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 80
NUM_SEEDS = 20
DROPOUT_RATE = 0.2

RESULTS_DIR = "results/"
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
SNN_DIR = os.path.join(RESULTS_DIR, "snn")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")
DATA_DIR = "data/mouse/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def load_gene_embeddings(batch_id, noise_factor):
    """Load CAE gene embeddings for a specific batch."""
    csv_file_path = os.path.join(EMBEDDINGS_DIR, 
                                f"{batch_id}_noise{noise_factor}_gene_embeddings_noisy.csv")
    
    if not os.path.exists(csv_file_path):
        print(f"Warning: Embeddings not found at {csv_file_path}")
        return None
    
    return pd.read_csv(csv_file_path, index_col=0)

def load_loocv_data(batch_id, seed):
    """Load LOOCV data splits."""
    csv_path = os.path.join(DATA_DIR, 'LOOCV_splits', f'imbalanced_x{NEG_POS_RATIO}',
                           f'naive_split_full_table_seed{seed}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: LOOCV data not found at {csv_path}")
        return None
    
    label_df = pd.read_csv(csv_path, index_col=0)
    label_df = label_df.drop(columns=['cv_fold'] if 'cv_fold' in label_df.columns else [])
    label_df = label_df.rename(columns={
        'ligand': 'Ligand', 'receptor': 'Receptor', 'y_label': 'Label', 'GPCR': 'GPCR_Index'})
    
    return label_df

def create_pairs(gene_embeddings, gene_pairs_df):
    """Create paired embeddings for ligand-receptor pairs."""
    embeddings = gene_embeddings.astype(np.float32)
    
    # Check for missing gene pairs
    missing_ligands = gene_pairs_df[~gene_pairs_df['Ligand'].isin(embeddings.index)]
    missing_receptors = gene_pairs_df[~gene_pairs_df['Receptor'].isin(embeddings.index)]
    
    if len(missing_ligands) > 0 or len(missing_receptors) > 0:
        print(f"  Warning: Missing genes - Ligands: {len(missing_ligands)}, Receptors: {len(missing_receptors)}")
    
    gene_pairs_df = gene_pairs_df[
        gene_pairs_df['Ligand'].isin(embeddings.index) & 
        gene_pairs_df['Receptor'].isin(embeddings.index)]
    
    # Vectorized operations
    ligand_embeddings = embeddings.loc[gene_pairs_df['Ligand']].values
    receptor_embeddings = embeddings.loc[gene_pairs_df['Receptor']].values
    
    all_pairs = np.concatenate([ligand_embeddings, receptor_embeddings], axis=1)
    all_labels = np.where(gene_pairs_df['Label'] == 1., 1., 0.)
    all_is_gpcr = gene_pairs_df['GPCR_Index'].values
    all_receptors = gene_pairs_df['Receptor'].values

    return (all_pairs.astype(np.float32),
            all_labels.astype(np.float32),
            all_is_gpcr.astype(np.float32),
            all_receptors,
            gene_pairs_df[['Ligand', 'Receptor']])

class GeneEmbeddingDataset(Dataset):
    """Dataset for gene embedding pairs."""
    
    def __init__(self, pairs, labels, is_gpcr=None):
        self.pairs = torch.FloatTensor(pairs)
        self.labels = torch.FloatTensor(labels)
        self.is_gpcr = is_gpcr

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'pair': self.pairs[idx],
            'label': self.labels[idx]
        }
        if self.is_gpcr is not None:
            sample['is_gpcr'] = self.is_gpcr[idx]
        return sample

class ImbalancedBatchSampler(Sampler):
    """Sampler for handling imbalanced datasets."""
    
    def __init__(self, labels, batch_size, pos_neg_ratio=1/10):
        self.labels = labels
        self.batch_size = batch_size
        self.pos_neg_ratio = pos_neg_ratio
        
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        
        self.pos_per_batch = int(batch_size * pos_neg_ratio)
        self.neg_per_batch = batch_size - self.pos_per_batch
        
    def __iter__(self):
        pos_indices = np.random.permutation(self.pos_indices)
        neg_indices = np.random.permutation(self.neg_indices)
        
        pos_batches = [pos_indices[i:i + self.pos_per_batch] 
                      for i in range(0, len(pos_indices), self.pos_per_batch)]
        neg_batches = [neg_indices[i:i + self.neg_per_batch] 
                      for i in range(0, len(neg_indices), self.neg_per_batch)]
        
        num_batches = min(len(pos_batches), len(neg_batches))

        for i in range(num_batches):
            yield np.concatenate([pos_batches[i], neg_batches[i]])

    def __len__(self):
        return min(len(self.pos_indices) // self.pos_per_batch,
                   len(self.neg_indices) // self.neg_per_batch)

class SiameseNetwork(nn.Module):
    """Siamese network for ligand-receptor prediction."""
    
    def __init__(self, input_shape, dropout_rate=0.2):
        super(SiameseNetwork, self).__init__()
        self.base_dropout_rate = dropout_rate
        
        self.base_network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.base_dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.base_dropout_rate),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[0] // 4), 64),
            nn.ReLU(),
            nn.Dropout(self.base_dropout_rate)
        )
        
        self.final_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        batch_size, features, _ = x.size()
        mid_point = features // 2
        
        x1, x2 = x[:, :mid_point, :], x[:, mid_point:, :]
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        
        out1 = self.base_network(x1)
        out2 = self.base_network(x2)
        
        merged = torch.cat((out1, out2), dim=1)
        return self.final_layers(merged)

class FZDPresenceError(Exception):
    """Custom exception for unexpected presence of FZD genes."""
    pass

def split_train_validation(receptor_train, train_ratio=0.8, seed=42, fzd_approach='include'):
    """Split training data with FZD gene handling."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    rng = np.random.default_rng(seed)
    unique_receptors = np.unique(receptor_train)
    rng.shuffle(unique_receptors)

    num_train = int(len(unique_receptors) * train_ratio)
    train_receptors = unique_receptors[:num_train]
    validation_receptors = unique_receptors[num_train:]
    
    # Handle FZD genes (mouse uses lowercase 'Fzd')
    if fzd_approach == 'exclude_val':
        print("  Excluding Fzd genes from validation set.") 
        original_count = len(validation_receptors)
        validation_receptors = validation_receptors[~np.char.startswith(validation_receptors.astype(str), 'Fzd')]
        removed_count = original_count - len(validation_receptors)
        print(f"  Removed {removed_count} Fzd receptors from validation")
    
    train_mask = np.isin(receptor_train, train_receptors)
    validation_mask = np.isin(receptor_train, validation_receptors)
    train_indices = np.where(train_mask)[0]
    validation_indices = np.where(validation_mask)[0]
    
    # Check for FZD presence
    def check_fzd_presence(receptors, set_name):
        fzd_receptors = [r for r in receptors if str(r).startswith('Fzd')]
        if fzd_receptors:
            logger.warning(f"Fzd genes found in {set_name} set: {fzd_receptors}")
            if fzd_approach in ['exclude', 'exclude_val'] and set_name == 'validation':
                raise FZDPresenceError(f"Fzd genes unexpectedly found in validation set: {fzd_receptors}")
            if fzd_approach == 'exclude' and set_name == 'training':
                raise FZDPresenceError(f"Fzd genes unexpectedly found in training set: {fzd_receptors}")
        else:
            logger.info(f"No Fzd genes found in {set_name} set.")

    if fzd_approach in ['exclude', 'exclude_val']:
        check_fzd_presence(validation_receptors, 'validation')
        if fzd_approach == 'exclude':
            check_fzd_presence(train_receptors, 'training')

    overlap_check = set(train_receptors).intersection(set(validation_receptors))
    if overlap_check:
        logger.warning(f"Overlap detected between training and validation receptors: {overlap_check}")

    logger.info(f"Unique receptors - Training: {len(train_receptors)}, Validation: {len(validation_receptors)}")

    return train_indices, validation_indices

def calculate_hypergeometric_pval(predictions, true_labels, k=20):
    """Calculate hypergeometric p-value for top-k predictions."""
    sorted_indices = np.argsort(predictions)[::-1]
    top_k_indices = sorted_indices[:k]
    
    N = len(predictions)  # Total population
    n = sum(true_labels)  # Total positives
    x = sum(true_labels[top_k_indices])  # Positives in top k
    
    # P(X >= x)
    return hypergeom.sf(x-1, N, n, k)

def create_test_data_for_receptor(receptor, all_possible_ligands, label_df):
    """Create test data for a specific receptor."""
    test_data = []
    gpcr_index = label_df[label_df['Receptor'] == receptor]['GPCR_Index'].iloc[0]
    
    for ligand in all_possible_ligands:
        y_label = label_df[(label_df['Ligand'] == ligand) & 
                          (label_df['Receptor'] == receptor)]['Label'].values
        y_label = 1 if (len(y_label) > 0 and y_label[0] == 1) else 0
        test_data.append({
            'Ligand': ligand, 
            'Receptor': receptor, 
            'Label': y_label, 
            'GPCR_Index': gpcr_index
        })
    
    return pd.DataFrame(test_data)

def plot_training_curves(history, receptor, output_dir):
    """Plot training curves for a receptor."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train', color='blue')
    ax1.plot(history['val_loss'], label='Val', color='orange')
    ax1.set_title(f'Loss - {receptor}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUPRC
    ax2.plot(history['train_auprc'], label='Train', color='blue')
    ax2.plot(history['val_auprc'], label='Val', color='orange')
    ax2.set_title(f'AUPRC - {receptor}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUPRC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AUROC
    ax3.plot(history['train_auroc'], label='Train', color='blue')
    ax3.plot(history['val_auroc'], label='Val', color='orange')
    ax3.set_title(f'AUROC - {receptor}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_{receptor}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def train_siamese_model(train_loader, val_loader, input_shape, receptor, output_dir):
    """Train the Siamese network for LOOCV."""
    model = SiameseNetwork((input_shape[0],), dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_auprc = 0
    epochs_no_improve = 0
    early_stopping_patience = 10
    delay_epochs = 20
    
    history = {'train_loss': [], 'val_loss': [], 'train_auroc': [], 
               'val_auroc': [], 'train_auprc': [], 'val_auprc': []}

    scaler = GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for batch in train_loader:
            pairs, labels = batch['pair'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(pairs).squeeze(1)
                loss = criterion(outputs, labels).mean()
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_predictions.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        if len(train_labels) > 0 and len(np.unique(train_labels)) > 1:
            train_auroc = roc_auc_score(train_labels, train_predictions)
            train_auprc = average_precision_score(train_labels, train_predictions)
        else:
            train_auroc = train_auprc = 0.5
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                pairs, labels = batch['pair'].to(DEVICE), batch['label'].to(DEVICE)
                outputs = model(pairs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        if len(val_labels) > 0 and len(np.unique(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_predictions)
            val_auprc = average_precision_score(val_labels, val_predictions)
        else:
            val_auroc = val_auprc = 0.5
            
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)
        history['train_auprc'].append(train_auprc)
        history['val_auprc'].append(val_auprc)

        # Model checkpointing
        if val_auprc > best_val_auprc and epoch >= delay_epochs:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 
                      os.path.join(checkpoint_dir, f"symmSNN_receptor{receptor}.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve == early_stopping_patience and epoch >= delay_epochs:
            break
    
    # Load best model
    best_model = SiameseNetwork((input_shape[0],), dropout_rate=DROPOUT_RATE).to(DEVICE)
    checkpoint_path = os.path.join(checkpoint_dir, f"symmSNN_receptor{receptor}.pt")
    if os.path.exists(checkpoint_path):
        best_model.load_state_dict(torch.load(checkpoint_path))
    
    return best_model, history

def preprocess_data_for_receptor(train_data, test_data, embedding_df, fzd_approach):
    """Preprocess data for a specific receptor."""
    X_train_raw, y_train_raw, _, receptor_train_raw, train_pairs = create_pairs(
        embedding_df, train_data)
    X_test, y_test, _, receptor_test, test_pairs = create_pairs(
        embedding_df, test_data)

    train_indices, validation_indices = split_train_validation(
        receptor_train_raw, fzd_approach=fzd_approach)
    
    X_train, y_train = X_train_raw[train_indices], y_train_raw[train_indices]
    X_validation, y_validation = X_train_raw[validation_indices], y_train_raw[validation_indices]
    
    # Reshape for CNN
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_validation_reshaped = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return (X_train_reshaped, y_train, X_validation_reshaped, y_validation, 
            X_test_reshaped, y_test, test_pairs)

def train_and_evaluate_loocv(embedding_df, label_df, batch_id, output_dir, 
                           gpcr_only=True, fzd_approach='include'):
    """Train and evaluate LOOCV models."""
    # Get unique receptors
    unique_receptors = label_df['Receptor'].unique()
    if gpcr_only:
        unique_receptors = label_df[label_df['GPCR_Index'] == 1]['Receptor'].unique()
    else:
        unique_receptors = label_df[label_df['GPCR_Index'] == 0]['Receptor'].unique()
    
    # Handle FZD exclusion
    if fzd_approach in ['exclude', 'exclude_val']:
        unique_receptors = unique_receptors[~np.char.startswith(unique_receptors.astype(str), 'Fzd')]
    
    all_possible_ligands = label_df['Ligand'].unique()
    results = {}
    all_test_data = []
    
    print(f"\n  Processing {len(unique_receptors)} receptors for LOOCV...")
    
    for receptor_idx, receptor in enumerate(unique_receptors):
        print(f"\n  [{receptor_idx+1}/{len(unique_receptors)}] Training for receptor {receptor}...")
        
        # Create test data for this receptor
        test_data = create_test_data_for_receptor(receptor, all_possible_ligands, label_df)
        train_data = label_df[label_df['Receptor'] != receptor].copy()
        
        # Preprocess data
        try:
            (X_train, y_train, X_val, y_val, X_test, 
             y_test, test_pairs) = preprocess_data_for_receptor(
                train_data, test_data, embedding_df, fzd_approach)
        except FZDPresenceError as e:
            print(f"    Skipping {receptor}: {e}")
            continue
        
        # Create datasets and loaders
        train_dataset = GeneEmbeddingDataset(X_train, y_train)
        val_dataset = GeneEmbeddingDataset(X_val, y_val)
        test_dataset = GeneEmbeddingDataset(X_test, y_test)
        
        # Use imbalanced sampler for train/val
        train_sampler = ImbalancedBatchSampler(train_dataset.labels, BATCH_SIZE)
        val_sampler = ImbalancedBatchSampler(val_dataset.labels, BATCH_SIZE)
        
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
                                num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler,
                              num_workers=6, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Train model
        input_shape = (X_train.shape[1] // 2,)
        model, history = train_siamese_model(
            train_loader, val_loader, input_shape, receptor, output_dir)
        plot_training_curves(history, receptor, output_dir)
        
        model.eval()
        test_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                pairs = batch['pair'].to(DEVICE)
                outputs = model(pairs).squeeze()
                probabilities = torch.sigmoid(outputs)
                test_predictions.extend(probabilities.cpu().numpy())
        
        test_predictions = np.array(test_predictions)
        
        # Calculate metrics
        if len(np.unique(y_test)) > 1:
            auroc = roc_auc_score(y_test, test_predictions)
            auprc = average_precision_score(y_test, test_predictions)
        else:
            auroc = auprc = np.nan
            
        hypergeom_pval = calculate_hypergeometric_pval(test_predictions, y_test)
        
        results[receptor] = {
            'auroc': auroc, 'auprc': auprc, 'hypergeom_pval': hypergeom_pval}
        
        # Add predictions to test data
        test_data_copy = test_data.copy()
        test_data_copy['Predictions'] = np.nan
        for (ligand, receptor_name), prediction in zip(test_pairs.values, test_predictions):
            mask = (test_data_copy['Ligand'] == ligand) & (test_data_copy['Receptor'] == receptor_name)
            test_data_copy.loc[mask, 'Predictions'] = prediction
        
        all_test_data.append(test_data_copy)
        
        # Save intermediate results
        if (receptor_idx + 1) % 10 == 0:
            current_all_test_data = pd.concat(all_test_data, ignore_index=True)
            current_all_test_data.to_csv(
                os.path.join(output_dir, f'intermediate_test_data_{receptor_idx+1}.csv'), 
                index=False)
        
        del X_train, y_train, X_val, y_val, X_test, y_test
        del train_dataset, val_dataset, test_dataset
        del train_loader, val_loader, test_loader
        del model, history
        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate all test data
    final_test_data = pd.concat(all_test_data, ignore_index=True)
    
    return results, final_test_data

def train_loocv_for_batch(batch_id, gpcr_only=True, fzd_approach='include'):
    """Train LOOCV for a specific batch."""
    print(f"\n{'='*60}")
    print(f"Training LOOCV for {batch_id}")
    print(f"GPCR only: {gpcr_only}, FZD approach: {fzd_approach}")
    print(f"{'='*60}")
    
    embedding_df = load_gene_embeddings(batch_id, NOISE_FACTOR)
    if embedding_df is None:
        print(f"Skipping {batch_id} - embeddings not found")
        return None
    
    receptor_type = "GPCR" if gpcr_only else "nonGPCR"
    fzd_prefix = ""
    if fzd_approach == 'exclude':
        fzd_prefix = "noFZD_"
    elif fzd_approach == 'exclude_val':
        fzd_prefix = "noFZDinVal_"
    
    result_id = f"loocv_{receptor_type}_{fzd_prefix}{batch_id}_dim{LATENT_DIM}_noise{NOISE_FACTOR}"
    output_base_dir = os.path.join(SNN_DIR, result_id)
    
    all_results = {}
    
    for seed in range(1, NUM_SEEDS + 1):
        print(f"\n  Running seed {seed}/{NUM_SEEDS}...")
        
        label_df = load_loocv_data(batch_id, seed)
        if label_df is None:
            print(f"  Skipping seed {seed} - data not found")
            continue
        
        if fzd_approach == 'exclude':
            print(f"    Filtering out Fzd receptors...")
            original_len = len(label_df)
            label_df = label_df[~label_df['Receptor'].str.startswith('Fzd')].copy()
            print(f"    Filtered from {original_len} to {len(label_df)} pairs")
        
        output_dir = os.path.join(output_base_dir, f'seed{seed}')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Train and evaluate
            results, all_test_data = train_and_evaluate_loocv(
                embedding_df, label_df, batch_id, output_dir, 
                gpcr_only=gpcr_only, fzd_approach=fzd_approach)
            
            all_results[seed] = {
                "results": results, "label_df": all_test_data}
            
            # Save results for this seed
            with open(os.path.join(output_dir, f'results_seed{seed}.pickle'), 'wb') as f:
                pickle.dump(all_results[seed], f)
                
        except Exception as e:
            print(f"    Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save all results
    with open(os.path.join(output_base_dir, 'all_results.pickle'), 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n  {batch_id} LOOCV complete! Results saved to {output_base_dir}")
    
    return all_results

def main():
    """Main LOOCV training pipeline."""
    print("="*80)
    print("Mouse Brain SNN LOOCV Training Pipeline")
    print("="*80)
    
    set_seed(42)
    
    print(f"\nUsing device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    args = parse_arguments()
    # latent_dim = args.latent_dim
    # neg_pos_ratio = args.neg_pos_ratio
    # noise_factor = args.noise_factor
    # dataset_index = args.dataset_index
    # weight_decay = args.weight_decay
    gpcr_only = args.gpcr_only
    fzd_approach = args.fzd_approach

    os.makedirs(SNN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Train for each batch
    start_time = time.time()
    all_batch_results = {}
    
    for batch_id in BATCHES:
        try:
            batch_results = train_loocv_for_batch(batch_id, gpcr_only, fzd_approach)
            if batch_results is not None:
                all_batch_results[batch_id] = batch_results
        except Exception as e:
            print(f"\nError training {batch_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*80}")
    print(f"LOOCV training complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {SNN_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()