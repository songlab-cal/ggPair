#!/usr/bin/env python
"""
Prepare ligand-receptor data splits for training and evaluation.

This script:
1. Creates 5-fold cross-validation splits (balanced and imbalanced)
2. Creates LOOCV splits for leave-one-out cross-validation
3. Handles both human and mouse datasets

Data structure:
- Human: Datasets 7, 23, 24
- Mouse: Combined mouse brain data
"""

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# Configuration
RANDOM_SEED = 0
NUM_SEEDS = 20
NUM_FOLDS = 5
IMBALANCE_RATIO = 10  # negative:positive ratio

# Datasets
HUMAN_DATASETS = [7, 23, 24]
MOUSE_DATASET = 'mouse_brain'

def set_seed(seed_value=0):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)

def build_hLR_dict(LR_pairs, gene_list):
    """Map receptor -> ligands from LR_pairs."""
    h_LR = defaultdict(list)
    for _, row in LR_pairs.iterrows():
        ligand = row['ligand_name']
        receptor = row['receptor_name']
        if ligand in gene_list and receptor in gene_list:
            h_LR[receptor].append(ligand)
    return h_LR

def generate_LR_pairs(h_LR_original, sub_ligand_list, sub_receptor_list, count=None, times=1):
    """Generate positive and negative ligand-receptor pairs."""
    h_LR = defaultdict(list)
    for receptor in h_LR_original.keys():
        if receptor in sub_receptor_list:
            for ligand in h_LR_original[receptor]:
                if ligand in sub_ligand_list:
                    h_LR[receptor].append(ligand)
    
    if count is None:
        count = 0
    
    gene_pair_list = []
    Y_data = []
    sub_receptor_list_ovlp = list(h_LR.keys())
    
    for receptor in sub_receptor_list_ovlp:
        non_pair_ligand_list = [i for i in sub_ligand_list if i not in h_LR[receptor]]
        random.seed(count)
        random.shuffle(non_pair_ligand_list)
        
        for ligand in h_LR[receptor]:
            # Positive pair
            gene_pair_list.append(ligand + '\t' + receptor)
            Y_data.append(1)
            
            # Negative pairs
            for _ in range(times):
                if non_pair_ligand_list:
                    random_ligand = non_pair_ligand_list.pop()
                    gene_pair_list.append(random_ligand + '\t' + receptor)
                    Y_data.append(0)
            count += 1
    
    Y_data_array = np.array(Y_data)
    gene_pair_list_array = np.array(gene_pair_list)
    
    return Y_data_array, gene_pair_list_array

def cross_validate_datasplit(LR_pairs, GPCR_list, output_dir, times=1, balanced=True, 
                           total_seeds=20, total_folds=5):
    """
    Perform 5-fold cross-validation on LR_pairs and save results.
    """
    # Get unique ligands and receptors
    ligand_list = pd.DataFrame(LR_pairs['ligand_name'].unique(), columns=['ligand_name'])
    receptor_list = pd.DataFrame(LR_pairs['receptor_name'].unique(), columns=['receptor_name'])
    gene_list = pd.concat([ligand_list['ligand_name'], receptor_list['receptor_name']]).unique().tolist()

    ovlp_ligand_list = [g for g in gene_list if g in ligand_list['ligand_name'].values]
    ovlp_receptor_list = [g for g in gene_list if g in receptor_list['receptor_name'].values]
    h_LR = build_hLR_dict(LR_pairs, gene_list)

    print(f"  Total ligands: {len(ovlp_ligand_list)}")
    print(f"  Total receptors: {len(ovlp_receptor_list)}")
    print(f"  Total positive pairs: {sum(len(v) for v in h_LR.values())}")

    for seed in range(1, total_seeds + 1):
        random.seed(seed)
        shuffled_receptors = random.sample(ovlp_receptor_list, len(ovlp_receptor_list))

        for fold in range(1, total_folds + 1):
            total = len(shuffled_receptors)
            start = int(np.ceil((fold - 1) * total / total_folds))
            end = int(np.ceil(fold * total / total_folds))

            test_receptors = shuffled_receptors[start:end]
            train_receptors = [r for r in shuffled_receptors if r not in test_receptors]

            # Get ligand indices
            test_ligands = LR_pairs['ligand_name'][LR_pairs['receptor_name'].isin(test_receptors)]
            test_ligand_idx = np.where(np.isin(ovlp_ligand_list, test_ligands))[0].tolist()
            train_ligands = LR_pairs['ligand_name'][LR_pairs['receptor_name'].isin(train_receptors)]
            train_ligand_idx = np.where(np.isin(ovlp_ligand_list, train_ligands))[0].tolist()
            
            if balanced:
                train_ligand_subset = np.array(ovlp_ligand_list)[train_ligand_idx]
                test_ligand_subset = np.array(ovlp_ligand_list)[test_ligand_idx]
            else:
                train_ligand_subset = np.array(ovlp_ligand_list)
                test_ligand_subset = np.array(ovlp_ligand_list)

            # Generate train/test sets
            Y_train, gene_pairs_train = generate_LR_pairs(
                h_LR, train_ligand_subset,
                np.array(ovlp_receptor_list)[[ovlp_receptor_list.index(r) for r in train_receptors]],
                times=times, count=(seed * fold if not balanced else None))
            Y_test, gene_pairs_test = generate_LR_pairs(
                h_LR, test_ligand_subset,
                np.array(ovlp_receptor_list)[[ovlp_receptor_list.index(r) for r in test_receptors]],
                times=times, count=(seed * fold if not balanced else None))

            split_dir = os.path.join(output_dir, f"seed{seed}")
            os.makedirs(split_dir, exist_ok=True)

            np.save(os.path.join(split_dir, f"{fold}_train_Y_data_array.npy"), Y_train)
            np.save(os.path.join(split_dir, f"{fold}_train_gene_pair_list_array.npy"), gene_pairs_train)
            np.save(os.path.join(split_dir, f"{fold}_test_Y_data_array.npy"), Y_test)
            np.save(os.path.join(split_dir, f"{fold}_test_gene_pair_list_array.npy"), gene_pairs_test)

            # GPCR annotation
            def annotate_gpcr(pairs):
                df = pd.DataFrame([p.split("\t") for p in pairs], columns=["ligand", "receptor"])
                return df['receptor'].isin(GPCR_list).astype(int).to_numpy()

            np.save(os.path.join(split_dir, f"{fold}_test_gene_pair_gpcr_index_array.npy"),
                    annotate_gpcr(gene_pairs_test))
            np.save(os.path.join(split_dir, f"{fold}_train_gene_pair_gpcr_index_array.npy"),
                    annotate_gpcr(gene_pairs_train))

def loocv_datasets(LR_pairs, GPCR_list, output_dir, times=1, total_seeds=20):
    """
    Generate LOOCV (Leave-One-Out Cross-Validation) datasets.
    Saves one CSV file per seed with columns: ligand, receptor, y_label, GPCR
    """
    # Get gene lists
    ligand_list = pd.DataFrame(LR_pairs['ligand_name'].unique(), columns=['ligand_name'])
    receptor_list = pd.DataFrame(LR_pairs['receptor_name'].unique(), columns=['receptor_name'])
    gene_list = np.concatenate([ligand_list['ligand_name'], receptor_list['receptor_name']])

    ovlp_ligand_list = [i for i in gene_list if i in list(ligand_list.iloc[:, 0])]
    ovlp_receptor_list = [i for i in gene_list if i in list(receptor_list.iloc[:, 0])]
    h_LR = build_hLR_dict(LR_pairs, gene_list)

    os.makedirs(output_dir, exist_ok=True)

    for count in range(1, total_seeds + 1):
        Y_data_array, gene_pair_list_array = generate_LR_pairs(
            h_LR, np.array(ovlp_ligand_list), np.array(ovlp_receptor_list),
            count=count, times=times)

        # Build DataFrame
        gene_pairs_tuples = [pair.split("\t") for pair in gene_pair_list_array.tolist()]
        gene_pairs_df = pd.DataFrame(gene_pairs_tuples, columns=["ligand", "receptor"])
        gene_pairs_df['y_label'] = Y_data_array

        # Annotate GPCR
        gene_pairs_df['GPCR'] = 0
        gene_pairs_df.loc[gene_pairs_df['receptor'].isin(GPCR_list), 'GPCR'] = 1

        out_file = os.path.join(output_dir, f'naive_split_full_table_seed{count}.csv')
        gene_pairs_df.to_csv(out_file, index=False)
        
    print(f"  Saved {total_seeds} LOOCV splits to {output_dir}")

def process_human_datasets(data_dir, output_base_dir):
    """Process all human datasets."""
    print("\nProcessing Human Datasets...")
    
    # Load GPCR list
    GPCR_df = pd.read_csv(os.path.join(data_dir, 'LR_databases', 'GPCRDB', 'GPCRTargets.csv'), header=1)
    GPCR_list = list(set(GPCR_df['HGNC symbol']))[1:]
    print(f"  Loaded {len(GPCR_list)} human GPCR genes")
    
    for dataset_number in HUMAN_DATASETS:
        print(f"\n  Dataset {dataset_number}:")
        
        # Load LR pairs
        LR_pairs = pd.read_csv(
            os.path.join(data_dir, 'LR_databases', 'outputs', 'filtered_datasets', 
                        'human', f'dataset{dataset_number}', 'filtered_pairs.csv'))
        
        # 5-fold CV - Balanced
        print("    Creating balanced 5-fold CV splits...")
        cross_validate_datasplit(
            LR_pairs, GPCR_list,
            output_dir=os.path.join(output_base_dir, '5foldcv_train_test_splits',
                                   f'balanced_dataset{dataset_number}'),
            balanced=True, total_seeds=NUM_SEEDS, total_folds=NUM_FOLDS)
        
        # 5-fold CV - Imbalanced
        print("    Creating imbalanced 5-fold CV splits...")
        cross_validate_datasplit(
            LR_pairs, GPCR_list,
            output_dir=os.path.join(output_base_dir, '5foldcv_train_test_splits',
                                   f'imbalanced_dataset{dataset_number}_x{IMBALANCE_RATIO}'),
            times=IMBALANCE_RATIO, balanced=False, total_seeds=NUM_SEEDS, total_folds=NUM_FOLDS)
        
        # LOOCV - Balanced
        print("    Creating balanced LOOCV splits...")
        loocv_datasets(
            LR_pairs, GPCR_list,
            output_dir=os.path.join(output_base_dir, 'LOOCV_splits',
                                   f'balanced_x1', f'dataset_{dataset_number}'))
        
        # LOOCV - Imbalanced
        print("    Creating imbalanced LOOCV splits...")
        loocv_datasets(
            LR_pairs, GPCR_list,
            output_dir=os.path.join(output_base_dir, 'LOOCV_splits',
                                   f'imbalanced_x{IMBALANCE_RATIO}', f'dataset_{dataset_number}'),
            times=IMBALANCE_RATIO)

def process_mouse_datasets(data_dir, output_base_dir):
    """Process mouse brain dataset."""
    print("\nProcessing Mouse Dataset...")
    
    # Load LR pairs
    LR_pairs = pd.read_csv(
        os.path.join(data_dir, 'LR_databases', 'outputs', 'filtered_datasets', 
                    'mouse_brain', 'filtered_pairs.csv'))
    
    # Load GPCR list (mouse)
    GPCR_df = pd.read_csv(os.path.join(data_dir, 'LR_databases', 'GPCRDB', 'GPCRTargets.csv'), header=1)
    GPCR_list = list(set(GPCR_df['MGI symbol']))[1:]
    print(f"  Loaded {len(GPCR_list)} mouse GPCR genes")
    
    # 5-fold CV - Balanced
    print("  Creating balanced 5-fold CV splits...")
    cross_validate_datasplit(
        LR_pairs, GPCR_list,
        output_dir=os.path.join(output_base_dir, 'mouse', '5foldcv_splits', 'balanced'),
        balanced=True, total_seeds=NUM_SEEDS, total_folds=NUM_FOLDS)
    
    # 5-fold CV - Imbalanced
    print("  Creating imbalanced 5-fold CV splits...")
    cross_validate_datasplit(
        LR_pairs, GPCR_list,
        output_dir=os.path.join(output_base_dir, 'mouse', '5foldcv_splits', 'imbalanced'),
        times=IMBALANCE_RATIO, balanced=False, total_seeds=NUM_SEEDS, total_folds=NUM_FOLDS)
    
    # LOOCV - Balanced
    print("  Creating balanced LOOCV splits...")
    loocv_datasets(
        LR_pairs, GPCR_list,
        output_dir=os.path.join(output_base_dir, 'mouse', 'LOOCV_splits', 'balanced'))
    
    # LOOCV - Imbalanced
    print("  Creating imbalanced LOOCV splits...")
    loocv_datasets(
        LR_pairs, GPCR_list,
        output_dir=os.path.join(output_base_dir, 'mouse', 'LOOCV_splits', f'imbalanced_x{IMBALANCE_RATIO}'),
        times=IMBALANCE_RATIO)

def main():
    """Main pipeline for creating train/test splits."""
    print("="*80)
    print("Creating Ligand-Receptor Data Splits")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    output_base_dir = data_dir  # Save splits in data directory
    
    # Check if LR database exists
    lr_db_path = os.path.join(data_dir, 'LR_databases')
    if not os.path.exists(lr_db_path):
        print(f"Error: LR database not found at {lr_db_path}")
        print("Please ensure ligand-receptor pairs are available.")
        return
    
    process_human_datasets(data_dir, output_base_dir)
    
    process_mouse_datasets(data_dir, output_base_dir)
    
    print("\n" + "="*80)
    print("Data split creation complete!")
    print(f"Splits saved to: {data_dir}")
    print("="*80)

if __name__ == "__main__":
    main()