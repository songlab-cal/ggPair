#!/usr/bin/env python
"""
Train Siamese Neural Network (SNN) using Leave-One-Out Cross-Validation (LOOCV).

LOOCV is performed where each receptor is left out in turn for testing,
while all other receptors are used for training.
"""

import os
import sys
import gc
import time
import argparse
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

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Configuration
DATASETS = [7, 23, 24]
LATENT_DIMS = {7: 1024, 23: 4096, 24: 4096}
BATCH_SIZE = 99
LEARNING_RATE = 5e-4
NUM_EPOCHS = 80
NUM_SEEDS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description="ggPair LOOCV training")
    parser.add_argument("--dataset", type=int, choices=DATASETS, help="Dataset index")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device to use")
    parser.add_argument("--neg-pos-ratio", type=int, default=10, help="Negative to positive ratio")
    parser.add_argument("--noise-factor", type=float, default=1.0, help="Noise factor")
    parser.add_argument("--gpcr-only", type=lambda x: str(x).lower() == 'true', default=True, 
                        help="Use only GPCR receptors")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--fzd-approach", type=str, choices=['include', 'exclude', 'exclude_val'], 
                        default='include', help="Approach for handling FZD genes")
    parser.add_argument("--all-datasets", action='store_true', help="Run all datasets")
    return parser.parse_args()

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class FZDPresenceError(Exception):
    """Custom exception for unexpected presence of FZD genes."""
    pass

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

def load_gene_embeddings(dataset_idx, noise_factor, project_root):
    """Load CAE gene embeddings."""
    csv_file_path = os.path.join(project_root, 'results', 'embeddings',
                                 f'dataset{dataset_idx}_noise{noise_factor}_gene_embeddings_noisy.csv')
    
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Embeddings not found: {csv_file_path}")
    
    return pd.read_csv(csv_file_path, index_col=0)

def load_loocv_data_for_seed(dataset_idx, seed, neg_pos_ratio, project_root):
    """Load LOOCV data split for a specific seed."""
    data_path = os.path.join(project_root, 'data', 'loocv_splits',
                            f'imbalanced_x{neg_pos_ratio}_dataset{dataset_idx}',
                            f'naive_split_full_table_seed{seed}.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"LOOCV data not found: {data_path}")
    
    label_df = pd.read_csv(data_path, index_col=0)
    label_df = label_df.drop(columns=['cv_fold'])
    label_df = label_df.rename(columns={
        'ligand': 'Ligand',
        'receptor': 'Receptor',
        'y_label': 'Label',
        'GPCR': 'GPCR_Index'
    })
    
    return label_df

def create_pairs(gene_embeddings, gene_pairs_df):
    """Create paired embeddings for ligand-receptor pairs."""
    embeddings = gene_embeddings.astype(np.float32)
    
    # Check for missing genes
    missing_ligands = gene_pairs_df[~gene_pairs_df['Ligand'].isin(embeddings.index)]
    missing_receptors = gene_pairs_df[~gene_pairs_df['Receptor'].isin(embeddings.index)]
    
    if len(missing_ligands) > 0 or len(missing_receptors) > 0:
        raise ValueError(f"Missing genes in embeddings!\n"
                        f"Missing ligands: {len(missing_ligands)}\n"
                        f"Missing receptors: {len(missing_receptors)}")
    
    # Filter to valid pairs
    gene_pairs_df = gene_pairs_df[
        gene_pairs_df['Ligand'].isin(embeddings.index) & 
        gene_pairs_df['Receptor'].isin(embeddings.index)
    ]
    
    # Create paired embeddings
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

def split_train_validation(receptor_train, train_ratio=0.8, seed=42, fzd_approach='include'):
    """Split training data into train/validation based on receptors."""
    logger = logging.getLogger(__name__)
    
    rng = np.random.default_rng(seed)
    unique_receptors = np.unique(receptor_train)
    rng.shuffle(unique_receptors)

    num_train = int(len(unique_receptors) * train_ratio)
    train_receptors = unique_receptors[:num_train]
    validation_receptors = unique_receptors[num_train:]
    
    if fzd_approach == 'exclude_val':
        validation_receptors = validation_receptors[
            ~np.char.startswith(validation_receptors.astype(str), 'FZD')
        ]
    
    train_mask = np.isin(receptor_train, train_receptors)
    validation_mask = np.isin(receptor_train, validation_receptors)
    
    train_indices = np.where(train_mask)[0]
    validation_indices = np.where(validation_mask)[0]
    
    # Check FZD presence
    def check_fzd_presence(receptors, set_name):
        fzd_receptors = [r for r in receptors if str(r).startswith('FZD')]
        if fzd_receptors:
            logger.warning(f"FZD genes found in {set_name} set: {fzd_receptors}")
            if fzd_approach in ['exclude', 'exclude_val'] and set_name == 'validation':
                raise FZDPresenceError(f"FZD genes unexpectedly found in validation set: {fzd_receptors}")
            if fzd_approach == 'exclude' and set_name == 'training':
                raise FZDPresenceError(f"FZD genes unexpectedly found in training set: {fzd_receptors}")
        else:
            logger.info(f"No FZD genes found in {set_name} set.")

    if fzd_approach in ['exclude', 'exclude_val']:
        check_fzd_presence(validation_receptors, 'validation')
        if fzd_approach == 'exclude':
            check_fzd_presence(train_receptors, 'training')

    logger.info(f"Unique receptors - Train: {len(train_receptors)}, Val: {len(validation_receptors)}")
    
    return train_indices, validation_indices

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

def train_siamese_model(train_loader, val_loader, input_shape, receptor, output_dir, 
                       learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS, 
                       weight_decay=1e-3, device=DEVICE):
    """Train the Siamese network."""
    model = SiameseNetwork((input_shape[0],)).to(device)
    
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_auprc = 0
    epochs_no_improve = 0
    early_stopping_patience = 10
    delay_epochs = 20
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_auprc': [], 'val_auprc': []
    }

    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        for batch in train_loader:
            pairs, labels = batch['pair'].to(device), batch['label'].to(device)
            
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
                pairs, labels = batch['pair'].to(device), batch['label'].to(device)
                outputs = model(pairs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_predictions.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_auroc = roc_auc_score(val_labels, val_predictions)
        val_auprc = average_precision_score(val_labels, val_predictions)
        val_loss /= len(val_loader)

        # Record history
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
    best_model = SiameseNetwork((input_shape[0],)).to(device)
    best_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, f"symmSNN_receptor{receptor}.pt"))
    )
    
    return best_model, history

def calculate_hypergeometric_pval(predictions, true_labels, k=20):
    """Calculate hypergeometric p-value for top-k predictions."""
    sorted_indices = np.argsort(predictions)[::-1]
    top_k_indices = sorted_indices[:k]
    
    N = len(predictions)
    n = sum(true_labels)
    x = sum(true_labels[top_k_indices])
    
    return hypergeom.sf(x-1, N, n, k)

def train_loocv_for_dataset(dataset_idx, args):
    """Train LOOCV for a specific dataset."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Training LOOCV for Dataset {dataset_idx}")
    logger.info(f"{'='*60}")
    
    # Get project root
    project_root = parent_dir
    latent_dim = LATENT_DIMS[dataset_idx]
    
    # Load embeddings
    logger.info(f"Loading embeddings (dim={latent_dim})...")
    embedding_df = load_gene_embeddings(dataset_idx, args.noise_factor, project_root)
    
    # Setup output directory
    if args.gpcr_only:
        receptor_type = "GPCR"
    else:
        receptor_type = "nonGPCR"
    
    fzd_prefix = ''
    if args.fzd_approach == 'exclude':
        fzd_prefix = 'noFZD_'
    elif args.fzd_approach == 'exclude_val':
        fzd_prefix = 'noFZDinVal_'
    
    result_id = (f"loocv_{receptor_type}_results/"
                f"{fzd_prefix}dataset{dataset_idx}_dim{latent_dim}_"
                f"noise{args.noise_factor}_imb{args.neg_pos_ratio}")
    
    output_base_dir = os.path.join(project_root, 'results', 'snn', result_id)
    
    # Process each seed
    all_results = {}
    
    for seed in range(1, NUM_SEEDS + 1):
        logger.info(f"\nProcessing seed {seed}/{NUM_SEEDS}...")
        
        try:
            # Load data
            label_df = load_loocv_data_for_seed(dataset_idx, seed, args.neg_pos_ratio, project_root)
            
            if args.fzd_approach == 'exclude':
                logger.info(f"Filtering FZD receptors...")
                original_len = len(label_df)
                label_df = label_df[~label_df['Receptor'].str.startswith('FZD')].copy()
                logger.info(f"Filtered {original_len - len(label_df)} rows")
            
            # Setup output directory for this seed
            output_dir = os.path.join(output_base_dir, f'seed{seed}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup logging for this seed
            seed_logger = setup_logging(output_dir)
            
            # Train and evaluate
            results, all_test_data = train_and_evaluate_model(
                embedding_df, label_df, output_dir, args
            )
            
            all_results[seed] = {
                "results": results,
                "label_df": all_test_data
            }
            
            # Save results for this seed
            with open(os.path.join(output_dir, f'results_seed{seed}.pickle'), 'wb') as f:
                pickle.dump(all_results[seed], f)
            
        except Exception as e:
            logger.error(f"Error in seed {seed}: {str(e)}")
            continue
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all results
    with open(os.path.join(output_base_dir, 'all_results.pickle'), 'wb') as f:
        pickle.dump(all_results, f)
    
    logger.info(f"\nDataset {dataset_idx} complete! Results saved to {output_base_dir}")
    
    return all_results

def train_and_evaluate_model(embedding_df, label_df, output_dir, args):
    """Train and evaluate model using LOOCV."""
    logger = logging.getLogger(__name__)
    
    # Filter receptors based on GPCR status
    if args.gpcr_only:
        unique_receptors = label_df[label_df['GPCR_Index'] == 1]['Receptor'].unique()
    else:
        unique_receptors = label_df[label_df['GPCR_Index'] == 0]['Receptor'].unique()
    
    # Filter FZD if needed
    if args.fzd_approach in ['exclude', 'exclude_val']:
        unique_receptors = unique_receptors[
            ~np.char.startswith(unique_receptors.astype(str), 'FZD')
        ]
    
    all_possible_ligands = label_df['Ligand'].unique()
    results = {}
    all_test_data = []
    
    logger.info(f"Total receptors to process: {len(unique_receptors)}")
    
    for i, receptor in enumerate(unique_receptors):
        logger.info(f"Training for receptor {receptor} ({i+1}/{len(unique_receptors)})...")
        
        # Create train/test split
        test_data = create_test_data_for_receptor(receptor, all_possible_ligands, label_df)
        train_data = label_df[label_df['Receptor'] != receptor].copy()
        
        # Preprocess data
        X_train_raw, y_train_raw, _, receptor_train_raw, train_pairs = create_pairs(
            embedding_df, train_data
        )
        X_test, y_test, _, receptor_test, test_pairs = create_pairs(
            embedding_df, test_data
        )
        
        # Split train into train/validation
        train_indices, val_indices = split_train_validation(
            receptor_train_raw, fzd_approach=args.fzd_approach
        )
        
        X_train = X_train_raw[train_indices]
        y_train = y_train_raw[train_indices]
        X_val = X_train_raw[val_indices]
        y_val = y_train_raw[val_indices]
        
        # Reshape for CNN
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Create datasets and loaders
        train_dataset = GeneEmbeddingDataset(X_train, y_train)
        val_dataset = GeneEmbeddingDataset(X_val, y_val)
        test_dataset = GeneEmbeddingDataset(X_test, y_test)
        
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
            train_loader, val_loader, input_shape, receptor, output_dir,
            weight_decay=args.weight_decay
        )
        
        # Evaluate model
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
        auroc = roc_auc_score(y_test, test_predictions)
        auprc = average_precision_score(y_test, test_predictions)
        hypergeom_pval = calculate_hypergeometric_pval(test_predictions, y_test)
        
        results[receptor] = {
            'auroc': auroc,
            'auprc': auprc,
            'hypergeom_pval': hypergeom_pval
        }
        
        logger.info(f"  AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, p-val: {hypergeom_pval:.4e}")
        
        # Add predictions to test data
        test_data_copy = test_data.copy()
        test_data_copy['Predictions'] = np.nan
        for (ligand, receptor_name), prediction in zip(test_pairs.values, test_predictions):
            mask = (test_data_copy['Ligand'] == ligand) & (test_data_copy['Receptor'] == receptor_name)
            test_data_copy.loc[mask, 'Predictions'] = prediction
        
        all_test_data.append(test_data_copy)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            current_test_data = pd.concat(all_test_data, ignore_index=True)
            current_test_data.to_csv(
                os.path.join(output_dir, f'intermediate_test_data_{i+1}.csv'), 
                index=False
            )
        
        # Clean up
        del model, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all test data
    final_test_data = pd.concat(all_test_data, ignore_index=True)
    
    # Verify all predictions are present
    if final_test_data['Predictions'].isna().any():
        logger.warning("Some predictions are missing!")
    
    return results, final_test_data

def main():
    """Main LOOCV pipeline."""
    args = parse_arguments()
    
    # Set GPU
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print("="*80)
    print("ggPair LOOCV Training Pipeline")
    print("="*80)
    
    # Set seed
    set_seed(42)
    
    # Check device
    print(f"\nUsing device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Process datasets
    start_time = time.time()
    
    if args.all_datasets:
        datasets_to_process = DATASETS
    else:
        datasets_to_process = [args.dataset] if args.dataset else DATASETS
    
    for dataset_idx in datasets_to_process:
        try:
            train_loocv_for_dataset(dataset_idx, args)
        except Exception as e:
            print(f"\nError processing dataset {dataset_idx}: {e}")
            continue
    
    # Report total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n{'='*80}")
    print(f"LOOCV training complete!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print(f"Results saved to: results/snn/loocv_*/")
    print("="*80)

if __name__ == "__main__":
    main()