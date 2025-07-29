#!/usr/bin/env python
"""
Train Siamese Neural Network (SNN) for mouse ligand-receptor prediction.

This script:
1. Loads CAE embeddings for each batch
2. Trains SNN with 5-fold cross-validation
3. Handles class imbalance (1:10 positive:negative ratio)
4. Processes each batch separately

Note: Mouse gene names use lowercase convention (e.g., 'Fzd' not 'FZD')
"""

import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Sampler, Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
BATCHES = ['B02_D1', 'B02_E1', 'B03_C2', 'B03_D2']
LATENT_DIM = 256  # From embeddings
NEG_POS_RATIO = 10
NOISE_FACTOR = 0.2
BATCH_SIZE = 99  # 9 positive + 90 negative samples
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 80
NUM_SEEDS = 20
DROPOUT_RATE = 0.2

# Paths
RESULTS_DIR = "results/"
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, "embeddings")
SNN_DIR = os.path.join(RESULTS_DIR, "snn")
FIG_DIR = os.path.join(RESULTS_DIR, "fig")
DATA_DIR = "data/mouse/"

# Device
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

def load_data_for_seed_test(batch_id, seed, neg_pos_ratio):
    """Load train/test split for a specific seed."""
    seed_dir = os.path.join(DATA_DIR, 'imbalanced', f'seed{seed}')
    
    if not os.path.exists(seed_dir):
        print(f"Warning: Seed directory not found: {seed_dir}")
        return None
    
    prefixes = [f"{i}_{t}" for i in range(1, 6) for t in ['train', 'test']]
    data_frames = []
    
    for prefix in prefixes:
        file_path_list = os.path.join(seed_dir, f'{prefix}_gene_pair_list_array.npy')
        file_path_gpcr = os.path.join(seed_dir, f'{prefix}_gene_pair_gpcr_index_array.npy')
        file_path_label = os.path.join(seed_dir, f'{prefix}_Y_data_array.npy')
        
        if all(os.path.isfile(f) for f in [file_path_list, file_path_gpcr, file_path_label]):
            data_list = np.load(file_path_list)
            data_gpcr = np.load(file_path_gpcr)
            data_label = np.load(file_path_label)
            
            first_gene, second_gene, label, number, train_test, gpcr_index = [], [], [], [], [], []
            for idx, pair in enumerate(data_list):
                genes = pair.split('\t')
                first_gene.append(genes[0])
                second_gene.append(genes[1])
                label.append('positive' if data_label[idx] == 1 else 'negative')
                number.append(int(prefix.split('_')[0]))
                train_test.append(prefix.split('_')[1])
                gpcr_index.append(data_gpcr[idx])
                
            df = pd.DataFrame({
                'Ligand': first_gene,
                'Receptor': second_gene,
                'Label': label,
                'CV_fold': number,
                'Train/Test': train_test,
                'GPCR_Index': gpcr_index
            })
            data_frames.append(df)
    
    return pd.concat(data_frames, ignore_index=True) if data_frames else None

def create_pairs(gene_embeddings, gene_pairs_df):
    """Create paired embeddings for ligand-receptor pairs."""
    embeddings = gene_embeddings.astype(np.float32)
    
    # Check for missing genes
    missing_ligands = gene_pairs_df[~gene_pairs_df['Ligand'].isin(embeddings.index)]
    missing_receptors = gene_pairs_df[~gene_pairs_df['Receptor'].isin(embeddings.index)]
    
    if len(missing_ligands) > 0 or len(missing_receptors) > 0:
        print(f"  Warning: Missing genes - Ligands: {len(missing_ligands)}, Receptors: {len(missing_receptors)}")
    
    # Filter to valid pairs
    gene_pairs_df = gene_pairs_df[
        gene_pairs_df['Ligand'].isin(embeddings.index) & 
        gene_pairs_df['Receptor'].isin(embeddings.index)]
    
    # Create paired embeddings
    ligand_embeddings = embeddings.loc[gene_pairs_df['Ligand']].values
    receptor_embeddings = embeddings.loc[gene_pairs_df['Receptor']].values
    
    all_pairs = np.concatenate([ligand_embeddings, receptor_embeddings], axis=1)
    all_labels = np.where(gene_pairs_df['Label'] == "positive", 1., 0.)
    all_cv_folds = gene_pairs_df['CV_fold'].values
    all_train_test_splits = gene_pairs_df['Train/Test'].values
    all_is_gpcr = gene_pairs_df['GPCR_Index'].values
    all_receptors = gene_pairs_df['Receptor'].values

    return (all_pairs.astype(np.float32),
            all_labels.astype(np.float32),
            all_cv_folds,
            all_train_test_splits,
            all_is_gpcr.astype(np.float32),
            all_receptors)

class GeneEmbeddingDataset(Dataset):
    """Dataset for gene embedding pairs."""
    
    def __init__(self, pairs, labels):
        self.pairs = torch.FloatTensor(pairs)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'pair': self.pairs[idx],
            'label': self.labels[idx]
        }

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
        self.dropout_rate = dropout_rate
        
        self.base_network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(32, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_rate),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[0] // 4), 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
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

def split_train_validation(receptor_train, train_ratio=0.8, seed=42):
    """Split training data into train/validation based on receptors."""
    rng = np.random.default_rng(seed)
    unique_receptors = np.unique(receptor_train)
    rng.shuffle(unique_receptors)

    num_train = int(len(unique_receptors) * train_ratio)
    train_receptors = unique_receptors[:num_train]
    validation_receptors = unique_receptors[num_train:]
    
    train_mask = np.isin(receptor_train, train_receptors)
    validation_mask = np.isin(receptor_train, validation_receptors)
    
    train_indices = np.where(train_mask)[0]
    validation_indices = np.where(validation_mask)[0]
    
    print(f"    Unique receptors - Train: {len(train_receptors)}, Val: {len(validation_receptors)}")
    
    # Check for overlap
    overlap = set(train_receptors).intersection(set(validation_receptors))
    if overlap:
        print(f"    Warning: Overlap detected between train/val receptors")
    
    return train_indices, validation_indices

def plot_training_curves(history, fold, batch_id, output_dir):
    """Plot training curves for a fold."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Val', color='orange', linewidth=2)
    ax1.set_title(f'Loss - {batch_id} Fold {fold}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUPRC
    ax2.plot(history['train_auprc'], label='Train', color='blue', linewidth=2)
    ax2.plot(history['val_auprc'], label='Val', color='orange', linewidth=2)
    ax2.set_title(f'AUPRC - {batch_id} Fold {fold}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUPRC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # AUROC
    ax3.plot(history['train_auroc'], label='Train', color='blue', linewidth=2)
    ax3.plot(history['val_auroc'], label='Val', color='orange', linewidth=2)
    ax3.set_title(f'AUROC - {batch_id} Fold {fold}')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_fold{fold}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def train_siamese_model(train_loader, val_loader, input_shape, fold, output_dir):
    """Train the Siamese network."""
    model = SiameseNetwork((input_shape[0],), dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    pos_weight = torch.tensor([10.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_auprc = 0
    epochs_no_improve = 0
    early_stopping_patience = 10
    delay_epochs = 20
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_auprc': [], 'val_auprc': []}

    scaler = GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
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
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_auroc = roc_auc_score(train_labels, train_predictions)
        train_auprc = average_precision_score(train_labels, train_predictions)
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
        
        val_auroc = roc_auc_score(val_labels, val_predictions)
        val_auprc = average_precision_score(val_labels, val_predictions)
        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_auroc)
        history['val_auroc'].append(val_auroc)
        history['train_auprc'].append(train_auprc)
        history['val_auprc'].append(val_auprc)

        print(f"  Epoch {epoch+1}: Train AUPRC={train_auprc:.4f}, Val AUPRC={val_auprc:.4f}")
        
        # Model checkpointing
        if val_auprc > best_val_auprc and epoch >= delay_epochs:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), 
                      os.path.join(checkpoint_dir, f"symmSNN_fold{fold}.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve == early_stopping_patience and epoch >= delay_epochs:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    best_model = SiameseNetwork((input_shape[0],), dropout_rate=DROPOUT_RATE).to(DEVICE)
    best_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, f"symmSNN_fold{fold}.pt")))
    
    return best_model, history

def train_and_evaluate_model(pairs_reshaped, labels, cv_folds, train_test_splits, 
                           label_df, receptors, batch_id, output_dir):
    """Train and evaluate model with 5-fold CV."""
    input_shape = (pairs_reshaped.shape[1] // 2,)
    unique_folds = np.unique(cv_folds)
    all_predictions = np.empty((len(pairs_reshaped),), dtype=np.float32)
    all_predictions.fill(np.nan)
    
    print(f"\n  Training {len(unique_folds)} folds for {batch_id}...")
    
    for fold in unique_folds:
        print(f"\n  Processing fold {fold}...")
        
        # Get data for this fold
        train_raw_indices = np.where((cv_folds == fold) & (train_test_splits == 'train'))[0]
        test_indices = np.where((cv_folds == fold) & (train_test_splits == 'test'))[0]
        
        X_train_raw = pairs_reshaped[train_raw_indices]
        y_train_raw = labels[train_raw_indices]
        X_test = pairs_reshaped[test_indices]
        y_test = labels[test_indices]
        
        # Split train into train/validation based on receptors
        receptor_train = receptors[train_raw_indices]
        train_indices, val_indices = split_train_validation(receptor_train)
        
        X_train = X_train_raw[train_indices]
        y_train = y_train_raw[train_indices]
        X_val = X_train_raw[val_indices]
        y_val = y_train_raw[val_indices]
        
        # Create data loaders
        train_dataset = GeneEmbeddingDataset(X_train, y_train)
        val_dataset = GeneEmbeddingDataset(X_val, y_val)
        
        train_sampler = ImbalancedBatchSampler(train_dataset.labels, BATCH_SIZE)
        val_sampler = ImbalancedBatchSampler(val_dataset.labels, BATCH_SIZE)
        
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, 
                                 num_workers=6, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, 
                               num_workers=6, pin_memory=True)

        # Train model
        model, history = train_siamese_model(
            train_loader, val_loader, input_shape, fold, output_dir)
        
        plot_training_curves(history, fold, batch_id, output_dir)
        
        # Make predictions on all data for this fold
        all_data = np.concatenate([X_train_raw, X_test])
        all_labels_fold = np.concatenate([y_train_raw, y_test])
        all_dataset = GeneEmbeddingDataset(all_data, all_labels_fold)
        all_loader = DataLoader(
            all_dataset, batch_size=1024, shuffle=False, 
            pin_memory=True, num_workers=6)
        
        model.eval()
        fold_predictions = []
        
        with torch.no_grad():
            for batch in all_loader:
                pairs = batch['pair'].to(DEVICE)
                outputs = model(pairs).squeeze()
                probabilities = F.sigmoid(outputs)
                fold_predictions.extend(probabilities.cpu().numpy())
        
        all_indices = np.concatenate([train_raw_indices, test_indices])
        all_predictions[all_indices] = np.array(fold_predictions)
        
        del X_train, X_val, y_train, y_val, train_dataset, val_dataset
        del train_loader, val_loader
        torch.cuda.empty_cache()
    
    label_df['Predictions'] = all_predictions
    
    return model, label_df

def train_snn_for_batch(batch_id):
    """Train SNN for a specific batch."""
    print(f"\n{'='*60}")
    print(f"Training SNN for {batch_id}")
    print(f"{'='*60}")
    
    print(f"\nLoading embeddings...")
    embedding_df = load_gene_embeddings(batch_id, NOISE_FACTOR)
    if embedding_df is None:
        print(f"Skipping {batch_id} - embeddings not found")
        return None
    
    print(f"  Embeddings shape: {embedding_df.shape}")
    
    result_id = f"5foldcv_{batch_id}_dim{LATENT_DIM}_noise{NOISE_FACTOR}"
    output_base_dir = os.path.join(SNN_DIR, result_id)
    
    results = {}
    
    for seed in range(1, NUM_SEEDS + 1):
        print(f"\n  Running seed {seed}/{NUM_SEEDS}...")
        
        # Load data for this seed
        label_df = load_data_for_seed_test(batch_id, seed, NEG_POS_RATIO)
        if label_df is None:
            print(f"  Skipping seed {seed} - data not found")
            continue
        
        # Create pairs
        pairs, labels, cv_folds, train_test_splits, is_gpcr, receptors = create_pairs(
            embedding_df, label_df)
        pairs_reshaped = pairs.reshape(pairs.shape[0], pairs.shape[1], 1)
        
        print(f"    Total pairs: {len(pairs)}")
        print(f"    Positive pairs: {sum(labels == 1)} ({sum(labels == 1)/len(labels)*100:.1f}%)")
        
        # Setup output directory for this seed
        output_dir = os.path.join(output_base_dir, f'seed{seed}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Train and evaluate
        best_model, new_label_df = train_and_evaluate_model(
            pairs_reshaped, labels, cv_folds, train_test_splits, 
            label_df, receptors, batch_id, output_dir)
        
        results[seed] = {"label_df": new_label_df}
        
        # Save results for this seed
        with open(os.path.join(output_dir, f'results_seed{seed}.pickle'), 'wb') as f:
            pickle.dump(results[seed], f)
        
        del pairs, labels, cv_folds, train_test_splits, pairs_reshaped
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save all results
    with open(os.path.join(output_base_dir, f'{batch_id}_all_seeds_results.pickle'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n  {batch_id} complete! Results saved to {output_base_dir}")
    
    return results

def main():
    """Main training pipeline."""
    print("="*80)
    print("Mouse Brain SNN Training Pipeline (5-fold CV)")
    print("="*80)
    
    set_seed(42)
    
    print(f"\nUsing device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available, training will be very slow!")
    
    os.makedirs(SNN_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Train for each batch
    start_time = time.time()
    all_batch_results = {}
    
    for batch_id in BATCHES:
        try:
            batch_results = train_snn_for_batch(batch_id)
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
    print(f"SNN training complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {SNN_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()