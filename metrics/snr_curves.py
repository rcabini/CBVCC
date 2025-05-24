import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, balanced_accuracy_score
)

#---------------------------------------------------------------

def compute_score_by_snr(all_data, metric, max_snr=30, bins=15):
    metric.index.names = ['file_id']
    metric.drop(['N.TRACKS'], axis='columns', inplace=True)
    all_datac = all_data.merge(metric, left_on='file_id', right_index=True, how='left')

    metrics_dict = {'SNR': [], 'Model': [], 'Score': []}
    snr_counts, snr_bins = np.histogram(all_datac['SNR'][all_datac['SNR'] < max_snr], bins=bins)

    for column in all_datac.columns:
        if column not in ['gt', 'SNR', 'file_id']:
            for i in range(1, len(snr_bins)):
                subset = all_datac[(all_datac['SNR'] >= snr_bins[i - 1]) & (all_datac['SNR'] < snr_bins[i])]
                subset = subset[subset['SNR'] < max_snr]

                if len(subset) > 1 and len(np.unique(subset['gt'])) > 1:
                    predictions = (subset[column] >= 0.5).astype(int)
                    auci = roc_auc_score(subset['gt'], subset[column])
                    precision = precision_score(subset['gt'], predictions, zero_division=0)
                    recall = recall_score(subset['gt'], predictions)
                    balanced_acc = balanced_accuracy_score(subset['gt'], predictions)
                    score = round(0.4 * auci + 0.2 * (precision + recall + balanced_acc), 3)

                    metrics_dict['SNR'].append(snr_bins[i])
                    metrics_dict['Model'].append(column)
                    metrics_dict['Score'].append(score)

    return pd.DataFrame(metrics_dict), snr_bins

#---------------------------------------------------------------

def plot_score_vs_snr(all_data, quality_metric, style_dict, clean_name_fn, output_path='test_snr_with_rank.png'):
    metrics_df, snr_bins = compute_score_by_snr(all_data, quality_metric)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                   gridspec_kw={'height_ratios': [2, 0.2]}, sharex=False)

    for model_name in metrics_df['Model'].unique():
        subset = metrics_df[metrics_df['Model'] == model_name]
        label = clean_name_fn(model_name)
        style = style_dict.get(model_name, {'linestyle': '-', 'color': 'black'})

        ax1.plot(
            subset['SNR'] - 2,
            subset['Score'],
            label=label,
            linestyle=style['linestyle'],
            color=style['color'],
            marker='o',
            linewidth=1.5,
            markersize=5
        )

    ax1.set_ylabel('Score')
    ax1.set_title('Score as a function of SNR')
    ax1.legend()
    ax1.grid(True)

    average_scores = metrics_df.groupby(['SNR', 'Model'])['Score'].mean().reset_index()
    average_scores['Rank'] = average_scores.groupby('SNR')['Score'].rank(ascending=False, method='min')
    average_scores['Adjusted_Rank'] = 8 - average_scores['Rank']

    additional_rows = pd.DataFrame({
        'SNR': [snr_bins[4], snr_bins[8]] * len(metrics_df['Model'].unique()),
        'Model': np.tile(metrics_df['Model'].unique(), 2),
        'Score': [np.nan] * len(metrics_df['Model'].unique()) * 2,
        'Rank': [0] * len(metrics_df['Model'].unique()) * 2,
        'Adjusted_Rank': [0] * len(metrics_df['Model'].unique()) * 2
    })

    average_scores = pd.concat([average_scores, additional_rows], ignore_index=True)

    color_map = {model: style_dict.get(model, {'color': 'black'})['color'] for model in metrics_df['Model'].unique()}
    average_scores['Color'] = average_scores['Model'].map(color_map)

    sns.barplot(
        x='SNR',
        y='Adjusted_Rank',
        hue='Model',
        data=average_scores,
        ax=ax2,
        palette=color_map
    )

    ax2.set_xlabel('SNR')
    ax2.set_ylabel('Rank')
    ax2.yaxis.set_ticks([])
    ax2.legend().set_visible(False)
    labels = [f'{np.round(snr_bins[i], 1)}-{np.round(snr_bins[i+1], 1)}' for i in range(9)]
    ax2.set_xticklabels(labels)
    plt.setp(ax2.get_xticklabels(), rotation=0)
    plt.tight_layout(pad=-1.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

