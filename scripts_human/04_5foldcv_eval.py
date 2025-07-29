#!/usr/bin/env python
"""
Evaluate model performance using 5-fold cross-validation results.

Compares ggPair performance against baseline methods (GCNG, SpatialDM)
and generates publication-quality figures.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    from utils.human_five_fold_eval import (
        process_results, subsample_negatives, compute_metrics,
        prepare_violin_data, plot_metric, calculate_stats
    )
except ImportError:
    print("Error: Cannot import evaluation utilities.")
    print("Please ensure utils/five_fold_eval.py exists")
    sys.exit(1)

# Configuration
DATASETS = [7, 23, 24]
LATENT_DIMS = {7: 1024, 23: 4096, 24: 4096}
NOISE_FACTOR = 1.0

# Set global font sizes for publication
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_ggpair_results(dataset_idx, project_root):
    """Load ggPair SNN results."""
    result_id = f"5foldcv_dataset{dataset_idx}_dim{LATENT_DIMS[dataset_idx]}_noise{NOISE_FACTOR}"
    results_path = os.path.join(project_root, 'results', 'snn', result_id, 'all_seeds_results.pickle')
    
    if not os.path.exists(results_path):
        print(f"Warning: ggPair results not found for dataset {dataset_idx}")
        return None
    
    with open(results_path, 'rb') as handle:
        results = pickle.load(handle)
    
    print(f"Processing ggPair results for dataset {dataset_idx}...")
    return process_results(results, model='CAE+SNN', imbalance_train_but_balance_test=True)

def load_baseline_results(dataset_idx, method, project_root):
    """Load baseline method results (GCNG or SpatialDM)."""
    if method == 'GCNG':
        csv_path = os.path.join(project_root, 'data', 'baseline_results', 
                               f'GCNG_dataset{dataset_idx}_imbalanced_original.csv')
    else:  # SpatialDM
        csv_path = os.path.join(project_root, 'data', 'baseline_results',
                               f'SpatialDM_by_receptor_dataset_{dataset_idx}_imbalanced.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: {method} results not found for dataset {dataset_idx}")
        return None
    
    results = pd.read_csv(csv_path, index_col=0)
    aggregated_results = {}
    
    for seed in range(1, 21):
        print(f"  Processing {method} seed {seed}...")
        df_aggregated = results[(results['Seed'] == seed) & (results['Train/Test'] == 'test')].copy()
        df_aggregated = df_aggregated.drop(columns=['Seed', 'Train/Test'])
        df_aggregated['Label'] = df_aggregated['Label'].map({1: 'positive', 0: 'negative'})
        df_aggregated = subsample_negatives(df_aggregated)
        df_aggregated.reset_index(drop=True, inplace=True)
        
        metrics, per_receptor_metrics = compute_metrics(df_aggregated)
        aggregated_results[seed] = {
            'Metrics': metrics,
            'MeanProb_Aggregated': df_aggregated,
            'Per_Receptor_Metrics': per_receptor_metrics
        }
    
    return aggregated_results

def create_performance_plots(master_results, dataset_idx, output_dir):
    """Create performance comparison plots."""
    # Prepare data for plotting
    custom_labels = ['Fold', 'GPCR', 'Serine_Threonine_Kinase', 'Tyrosine_Kinase', 'Other']
    
    # Initialize DataFrames
    df_auprc_all = pd.DataFrame()
    df_auroc_all = pd.DataFrame()
    df_per_receptor_auprc_all = pd.DataFrame()
    df_per_receptor_auroc_all = pd.DataFrame()
    
    # Process each method
    for method, results in master_results.items():
        df_acc, df_auroc, df_auprc, df_pr_auroc, df_pr_auprc = prepare_violin_data(
            results, method)
        
        df_auprc_all = pd.concat([df_auprc_all, df_auprc], ignore_index=True)
        df_auroc_all = pd.concat([df_auroc_all, df_auroc], ignore_index=True)
        df_per_receptor_auprc_all = pd.concat([df_per_receptor_auprc_all, df_pr_auprc], ignore_index=True)
        df_per_receptor_auroc_all = pd.concat([df_per_receptor_auroc_all, df_pr_auroc], ignore_index=True)
    
    # Add method column
    for df in [df_auprc_all, df_auroc_all, df_per_receptor_auprc_all, df_per_receptor_auroc_all]:
        df['Method'] = df['Batch']
        df['Metric'] = pd.Categorical(df['Metric'], categories=custom_labels, ordered=True)
    
    # Create plots
    order = ['ggPair', 'GCNG', 'SpatialDM']
    
    # AUPRC plot
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    plot_metric(df_auprc_all, 'AUPRC', order, ax1, title="")
    ax1.set_ylabel('AUPRC', fontsize=20)
    ax1.legend().set_visible(False)
    
    # Adjust x-axis limits
    x_min, x_max = ax1.get_xlim()
    ax1.set_xlim(x_min, x_max + (x_max - x_min) * 0.07)
    
    plt.tight_layout()
    auprc_path = os.path.join(output_dir, f'dataset{dataset_idx}_auprc_comparison.png')
    plt.savefig(auprc_path, bbox_inches='tight', dpi=300, format='png', transparent=True)
    plt.close()
    
    # Per-receptor AUPRC plot
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    plot_metric(df_per_receptor_auprc_all, 'Per-Receptor AUPRC', order, ax2, title="")
    ax2.set_ylabel('Per-Receptor AUPRC', fontsize=20)
    ax2.legend().set_visible(False)
    plt.tight_layout()
    
    pr_auprc_path = os.path.join(output_dir, f'dataset{dataset_idx}_per_receptor_auprc_comparison.png')
    plt.savefig(pr_auprc_path, bbox_inches='tight', dpi=300, format='png', transparent=True)
    plt.close()
    
    print(f"  Saved plots to {output_dir}")
    
    # Calculate and save statistics
    stats_auprc = calculate_stats(df_auprc_all)
    stats_pr_auprc = calculate_stats(df_per_receptor_auprc_all)
    
    # Rank by median performance
    def rank_by_median(group):
        ranked = group.sort_values('median', ascending=False)
        ranked['rank'] = range(1, len(ranked) + 1)
        return ranked
    
    auprc_ranked = stats_auprc.groupby('Metric').apply(rank_by_median).reset_index(drop=True)
    pr_auprc_ranked = stats_pr_auprc.groupby('Metric').apply(rank_by_median).reset_index(drop=True)
    
    # Save statistics
    stats_path = os.path.join(output_dir, f'dataset{dataset_idx}_performance_stats.csv')
    auprc_ranked.to_csv(stats_path)
    print(f"  Saved performance statistics to {stats_path}")
    
    return auprc_ranked, pr_auprc_ranked

def evaluate_dataset(dataset_idx, project_root):
    """Evaluate all methods for a specific dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating Dataset {dataset_idx}")
    print(f"{'='*60}")
    
    # Setup output directory
    output_dir = os.path.join(project_root, 'results', 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results for all methods
    master_results = {}
    
    # Load ggPair results
    ggpair_results = load_ggpair_results(dataset_idx, project_root)
    if ggpair_results is not None:
        master_results['ggPair'] = ggpair_results
    
    # Load baseline results
    for method in ['GCNG', 'SpatialDM']:
        baseline_results = load_baseline_results(dataset_idx, method, project_root)
        if baseline_results is not None:
            master_results[method] = baseline_results
    
    if not master_results:
        print(f"No results found for dataset {dataset_idx}")
        return None
    
    # Create performance plots
    auprc_stats, pr_auprc_stats = create_performance_plots(
        master_results, dataset_idx, output_dir
    )
    
    # Print summary statistics
    print(f"\nAUPRC Performance Summary:")
    print(auprc_stats[auprc_stats['Metric'] == 'Fold'][['Method', 'median', 'mean', 'std', 'rank']])
    
    print(f"\nPer-Receptor AUPRC Performance Summary:")
    print(pr_auprc_stats[pr_auprc_stats['Metric'] == 'Fold'][['Method', 'median', 'mean', 'std', 'rank']])
    
    return master_results

def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("ggPair Model Evaluation Pipeline")
    print("="*80)
    
    # Get project root
    project_root = parent_dir
    
    # Evaluate each dataset
    all_results = {}
    
    for dataset_idx in DATASETS:
        try:
            results = evaluate_dataset(dataset_idx, project_root)
            if results is not None:
                all_results[dataset_idx] = results
        except Exception as e:
            print(f"\nError evaluating dataset {dataset_idx}: {e}")
            continue
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: results/evaluation/")
    print("="*80)

if __name__ == "__main__":
    main()