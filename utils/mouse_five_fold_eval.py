# nonGPCR receptors do not need to capitalize.
# process_results for mouse is different from human data version.

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

receptor_serine_threonine_kinases = pd.read_csv('data/Receptor_Serine_Threonine_Kinases.csv',sep=',')
receptor_serine_threonine_kinases_gene_name_lists = receptor_serine_threonine_kinases['Gene Names'].str.split()
receptor_serine_threonine_kinases_genes = [gene for sublist in receptor_serine_threonine_kinases_gene_name_lists for gene in sublist]
len(receptor_serine_threonine_kinases_genes) # receptor_stk 23

receptor_tyrosine_kinase_raw = pd.read_csv('data/Receptor_Tyrosine_Kinase.tsv',sep='\t')
receptor_tyrosine_kinase = receptor_tyrosine_kinase_raw[
    receptor_tyrosine_kinase_raw['Reviewed'] == "reviewed"]
receptor_tyrosine_kinase_gene_name_lists = receptor_tyrosine_kinase['Gene Names'].str.split()
receptor_tyrosine_kinase_genes = [gene for sublist in receptor_tyrosine_kinase_gene_name_lists for gene in sublist]
len(receptor_tyrosine_kinase_genes) # receptor_tk 250

receptor_tyrosine_phosphatases_raw = pd.read_csv('data/Receptor_Tyrosine_Phosphatases.tsv',sep='\t')
receptor_tyrosine_phosphatases = receptor_tyrosine_phosphatases_raw[
    receptor_tyrosine_phosphatases_raw['Reviewed'] == "reviewed"]

receptor_tyrosine_phosphatases_gene_name_lists = receptor_tyrosine_phosphatases['Gene Names'].str.split()
receptor_tyrosine_phosphatases_genes = [gene for sublist in receptor_tyrosine_phosphatases_gene_name_lists for gene in sublist]
len(receptor_tyrosine_phosphatases_genes) # receptor_tp 70

def categorize_receptor(row):
    receptor_name = row['Receptor']
    if row['GPCR_Index'] == 1:
        return 'GPCR'
    elif receptor_name in receptor_serine_threonine_kinases_genes:
        return 'Serine_Threonine_Kinase'
    elif receptor_name in receptor_tyrosine_kinase_genes:
        return 'Tyrosine_Kinase'
    elif receptor_name in receptor_tyrosine_phosphatases_genes:
       return 'Tyrosine_Phosphatase'
    else:
        return 'Other'
    
def subsample_negatives(df):
    # Initialize an empty DataFrame to store the subsampled data
    subsampled_df = pd.DataFrame(columns=df.columns)

    # Get unique CV folds
    cv_folds = df['CV_fold'].unique()

    for fold in cv_folds:
        fold_df = df[df['CV_fold'] == fold]
        
        positives = fold_df[fold_df['Label'] == 'positive']
        negatives = fold_df[fold_df['Label'] == 'negative']
        
        # Initialize a DataFrame to store fold specific subsampled data
        fold_subsampled_df = positives.copy()
        
        # Create a dictionary to keep track of negative counts needed per receptor
        receptor_neg_count = positives['Receptor'].value_counts().to_dict()
        
        for receptor, count in receptor_neg_count.items():
            receptor_negatives = negatives[negatives['Receptor'] == receptor]
            
            if len(receptor_negatives) >= count:
                receptor_sampled_negatives = receptor_negatives.sample(n=count, random_state=42)
            else:
                receptor_sampled_negatives = receptor_negatives  # If less negatives than needed, take all
            
            fold_subsampled_df = pd.concat([fold_subsampled_df, receptor_sampled_negatives])

        subsampled_df = pd.concat([subsampled_df, fold_subsampled_df])

    return subsampled_df

def compute_metrics(df_aggregated, min_positives=4, min_negatives=4):
    metrics = {}
    per_receptor_metrics = []
    
    # Add the new receptor category column
    df_aggregated['Receptor_Category'] = df_aggregated.apply(categorize_receptor, axis=1)
    
    for fold in df_aggregated['CV_fold'].unique():
        fold_data = df_aggregated[df_aggregated['CV_fold'] == fold]
        fold_metrics = {}
        category_metrics = {}
        
        y_true_fold = fold_data['Label'].apply(lambda x: 1 if x == 'positive' else 0).values
        y_score_fold = fold_data['Predictions'].values
        
        # Compute fold-level metrics
        fold_metrics['AUROC'] = roc_auc_score(y_true_fold, y_score_fold)
        fold_metrics['AUPRC'] = average_precision_score(y_true_fold, y_score_fold)
        fold_fpr, fold_tpr, fold_thresholds = roc_curve(y_true_fold, y_score_fold)
        fold_optimal_idx = np.argmin(np.sqrt(np.square(1-fold_tpr) + np.square(fold_fpr)))
        fold_optimal_threshold = fold_thresholds[fold_optimal_idx]
        fold_metrics['Accuracy'] = accuracy_score(y_true_fold, (y_score_fold > fold_optimal_threshold).astype(int))

        # Compute category-level metrics
        for category in fold_data['Receptor_Category'].unique():
            category_data = fold_data[fold_data['Receptor_Category'] == category]
            
            if len(category_data) > 0:
                y_true_category = category_data['Label'].apply(lambda x: 1 if x == 'positive' else 0).values
                y_score_category = category_data['Predictions'].values
                
                if len(np.unique(y_true_category)) > 1:  # Check if there's more than one class
                    category_metrics[f'{category}_AUROC'] = roc_auc_score(y_true_category, y_score_category)
                    category_metrics[f'{category}_AUPRC'] = average_precision_score(y_true_category, y_score_category)
                    category_metrics[f'{category}_Accuracy'] = accuracy_score(y_true_category, (y_score_category > fold_optimal_threshold).astype(int))
                else:
                    category_metrics[f'{category}_AUROC'] = np.nan
                    category_metrics[f'{category}_AUPRC'] = np.nan
                    category_metrics[f'{category}_Accuracy'] = np.nan
            else:
                category_metrics[f'{category}_AUROC'] = np.nan
                category_metrics[f'{category}_AUPRC'] = np.nan
                category_metrics[f'{category}_Accuracy'] = np.nan

        # Compute per-receptor metrics
        for receptor in fold_data['Receptor'].unique():
            receptor_data = fold_data[fold_data['Receptor'] == receptor].copy()
            positives = receptor_data[receptor_data['Label'] == 'positive']
            negatives = receptor_data[receptor_data['Label'] == 'negative']
            
            if len(positives) >= min_positives and len(negatives) >= min_negatives:
                y_true_receptor = receptor_data['Label'].apply(lambda x: 1 if x == 'positive' else 0).values
                y_score_receptor = receptor_data['Predictions'].values
                
                try:
                    auroc = roc_auc_score(y_true_receptor, y_score_receptor)
                    auprc = average_precision_score(y_true_receptor, y_score_receptor)
                    
                    per_receptor_metrics.append({
                        'Fold': fold,
                        'Receptor': receptor,
                        'AUROC': auroc,
                        'AUPRC': auprc,
                        'Num_Positives': len(positives),
                        'Num_Negatives': len(negatives),
                        'Receptor_Category': receptor_data['Receptor_Category'].iloc[0]
                    })
                except ValueError:
                    # This can happen if all samples are from one class
                    continue
        
        metrics[fold] = {
            'Fold_Metrics': fold_metrics,
            'Category_Metrics': category_metrics
        }
    
    return metrics, pd.DataFrame(per_receptor_metrics)

# process_results for mouse is different from human data version
def process_results(results, test_type='mismatched', new_ggPair = False, model='GCNG', method='MeanProb', min_positives=4, min_negatives=4, num_seeds=20):
    aggregated_results = {}

    for seed in range(1, num_seeds + 1):
        # print(f"Aggregating results for seed {seed}...")

        # Get the expanded_label_df for this seed (test set only)
        if model == 'GCNG':
            df = results[seed]['expanded_label_df'][
                results[seed]['expanded_label_df']['Train/Test'] == 'test'].copy()
        else:
            df = results[seed]['label_df'][
                results[seed]['label_df']['Train/Test'] == 'test'].copy()
            if 'Test_Type' in df.columns:
                df = df[df['Test_Type'] == test_type].copy() # test_type is pseudo or mismatched

        if not new_ggPair: # remove batch prefix
            df['Ligand'] = df['Ligand'].apply(lambda x: x.split('_')[2])
            df['Receptor'] = df['Receptor'].apply(lambda x: x.split('_')[2])

        # Modify here for per-fold aggregation
        if method == 'MeanProb':
            df_aggregated = df.groupby(
                ['Ligand', 'Receptor', 'Label', 'CV_fold', 'GPCR_Index'])['Predictions'].mean().reset_index()
        else:
            df_aggregated = df.groupby(
                ['Ligand', 'Receptor', 'Label', 'CV_fold', 'GPCR_Index'])['Predictions'].max().reset_index()
        
        if new_ggPair:
            df_aggregated = subsample_negatives(df_aggregated)
            df_aggregated.reset_index(drop=True, inplace=True)

        metrics, per_receptor_metrics = compute_metrics(df_aggregated, min_positives, min_negatives)    
        aggregated_results[seed] = {
            'Metrics': metrics,
            f"{method}_Aggregated": df_aggregated,
            'Per_Receptor_Metrics': per_receptor_metrics
        }

    return aggregated_results

# Function to process a single seed
def process_new_seed(seed, df_list):
    dfs = []
    for df, batch in df_list:
        if seed in df:
            temp_df = df[seed]['label_df'].copy()
            temp_df['batch'] = batch
            dfs.append(temp_df)
    return pd.concat(dfs, ignore_index=True)

# Helper function to concatenate DataFrames
def concat_dfs(key_name, aggregated_results, batch_id, column_name='Predictions'):
    combined_df = pd.DataFrame()
    for i in range(1, len(aggregated_results) + 1):
    # for i in range(1, 21):
        temp_df = pd.DataFrame(aggregated_results[i][key_name][column_name], columns=[column_name])
        temp_df['Type'] = f"{key_name} {i}"
        temp_df['Batch'] = batch_id
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    return combined_df


def prepare_violin_data(aggregated_results, batch_id, method='MeanProb'):
    print(f"Processing batch_id: {batch_id}, method: {method}")
    data_for_violin_acc, data_for_violin_auroc, data_for_violin_auprc = [], [], []
    data_for_violin_per_receptor_auroc, data_for_violin_per_receptor_auprc = [], []
    
    for seed in range(1, len(aggregated_results) + 1):
        for fold in aggregated_results[seed]['Metrics']:
            fold_metrics = aggregated_results[seed]['Metrics'][fold]['Fold_Metrics']
            category_metrics = aggregated_results[seed]['Metrics'][fold]['Category_Metrics']

            # Fold-specific metrics (i.e., unstratified)
            common_data = {'Method': method, 'Seed': seed, 'Fold': fold, 'Batch': batch_id}
            data_for_violin_acc.append({**common_data, 'Metric': 'Fold', 'Value': fold_metrics['Accuracy']})
            data_for_violin_auroc.append({**common_data, 'Metric': 'Fold', 'Value': fold_metrics['AUROC']})
            data_for_violin_auprc.append({**common_data, 'Metric': 'Fold', 'Value': fold_metrics['AUPRC']})

            # Category-specific metrics
            for category, metrics in category_metrics.items():
                category_parts = category.split('_')
                if len(category_parts) >= 2:
                    category_name = '_'.join(category_parts[:-1])  # Join all parts except the last one
                    metric_type = category_parts[-1]   # Last part is the metric type
                else:
                    print(f"Warning: Unexpected category format: {category}")
                    continue
                
                if metric_type == 'Accuracy':
                    data_for_violin_acc.append({**common_data, 'Metric': category_name, 'Value': metrics})
                elif metric_type == 'AUROC':
                    data_for_violin_auroc.append({**common_data, 'Metric': category_name, 'Value': metrics})
                elif metric_type == 'AUPRC':
                    data_for_violin_auprc.append({**common_data, 'Metric': category_name, 'Value': metrics})

        # Per-receptor metrics
        per_receptor_metrics = aggregated_results[seed]['Per_Receptor_Metrics']
        for _, row in per_receptor_metrics.iterrows():
            common_data = {
                'Method': method, 'Seed': seed, 'Fold': row['Fold'], 'Batch': batch_id, 'Receptor': row['Receptor'],
                'Num_Positives': row['Num_Positives'], 'Num_Negatives': row['Num_Negatives']
            }
            
            # Use 'Receptor_Category' instead of 'Category'
            category = row.get('Receptor_Category', 'Unknown')
            
            # AUROC
            data_for_violin_per_receptor_auroc.append({
                **common_data,
                'Metric': category,
                'Value': row['AUROC']
            })
            data_for_violin_per_receptor_auroc.append({
                **common_data,
                'Metric': 'Fold',
                'Value': row['AUROC']
            })
            
            # AUPRC
            data_for_violin_per_receptor_auprc.append({
                **common_data,
                'Metric': category,
                'Value': row['AUPRC']
            })
            data_for_violin_per_receptor_auprc.append({
                **common_data,
                'Metric': 'Fold',
                'Value': row['AUPRC']
            })

    return (pd.DataFrame(data_for_violin_acc), 
            pd.DataFrame(data_for_violin_auroc), 
            pd.DataFrame(data_for_violin_auprc),
            pd.DataFrame(data_for_violin_per_receptor_auroc),
            pd.DataFrame(data_for_violin_per_receptor_auprc))

def plot_metric(data, y_label, order, ax, title='Mouse: balanced Testing with real ligands'):
    distinct_colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Lime
        '#FFA500',  # Orange
        '#800080',  # Purple
        '#00FFFF',  # Cyan
        '#FF00FF',  # Magenta
        '#008000',  # Green
        '#FFC0CB',  # Pink
        '#A52A2A',  # Brown
        '#FFD700',  # Gold
        '#808080',  # Gray
        '#40E0D0',  # Turquoise
        '#FF6347',  # Tomato
        '#7FFF00',  # Chartreuse
        '#8A2BE2',  # BlueViolet
        '#00FA9A',  # MediumSpringGreen
        '#1E90FF',  # DodgerBlue
    ]

    # Create a color map based on the method name
    color_map = {method: color for method, color in zip(order, distinct_colors)}

    data = data.replace('Fold', 'Overall')

    sns.boxplot(x='Metric', y='Value', hue='Method_Batch', data=data, ax=ax, palette=color_map)
    # sns.violinplot(x='Metric', y='Value', hue='Method_Batch', data=data, ax=ax, palette=color_map)
    ax.axhline(0.5, color='red', linestyle='--')
    ax.set_ylabel(y_label)
    ax.set_xlabel(None)
    # ax.legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(fontsize = "x-small", loc = "upper right")
    if title:
        ax.set_title(title)
    # Get current tick locations and labels
    locs, labels = plt.xticks()
    # Format x-axis labels
    new_labels = []
    for label in labels:
        text = label.get_text()
        if '_' in text:
            text = text.replace('_', '\n')
        new_labels.append(text)
    
    plt.xticks(locs, new_labels, ha='center', fontsize='small')

# Helper function to calculate mean, median, and 95% CI
def calculate_stats(df, value_col='Value', group_cols=['Metric', 'Method_Batch']):
    stats_df = df.groupby(group_cols).agg(
        mean=(value_col, 'mean'),
        median=(value_col, 'median'),
        std=(value_col, 'std'),
        n=(value_col, 'count')
    ).reset_index()
    
    stats_df['CI_lower'] = stats_df['mean'] - 1.96 * (stats_df['std'] / np.sqrt(stats_df['n']))
    stats_df['CI_upper'] = stats_df['mean'] + 1.96 * (stats_df['std'] / np.sqrt(stats_df['n']))
    
    return stats_df