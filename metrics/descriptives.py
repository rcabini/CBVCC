import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os

#---------------------------------------------------------------

def plot_class_distribution(counts, val, tes, tra, output_path='./classes.png'):
    sns.set_style("whitegrid")

    counts['Dataset'] = 'Training'
    counts.loc[counts.index.isin(val.index), 'Dataset'] = 'Validation'
    counts.loc[counts.index.isin(tes.index), 'Dataset'] = 'Test'
    counts['gt'] = counts.index.map(lambda x: (
        val.loc[x, 'gt'] if x in val.index else
        (tes.loc[x, 'gt'] if x in tes.index else
        (tra.loc[x, 'gt'] if x in tra.index else None))
    )) 
                 
    grouped = counts.groupby(['Dataset', 'gt']).size().reset_index(name='file_count')
    custom_palette = {0: 'royalblue', 1: 'red'}
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(5, 5.1))
    ax = sns.barplot(
        data=grouped,
        x='Dataset',
        y='file_count',
        hue='gt',
        order=['Training', 'Validation', 'Test'],
        palette=custom_palette,
        alpha=0.7
    )
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=16, color='black',
                    xytext=(0, 6), textcoords='offset points')
        p.set_edgecolor('white')
        p.set_linewidth(1)

    plt.ylim([0, 135])
    plt.xlabel('Dataset', fontsize=18)
    plt.ylabel('Number of video-patches', fontsize=18)
    plt.title('Number of video-patches per class', fontsize=18)
    plt.legend(title='Classes', fontsize=16, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

#---------------------------------------------------------------

def plot_metric_distributions(counts, metric, val, tes, output_path='./metrics.png'):
    dataset_labels = {
        'Training': f'Training (n={len(counts)})',
        'Validation': f'Validation (n={len(val)})',
        'Test': f'Test (n={len(tes)})'
    }

    counts['Dataset'] = dataset_labels['Training']
    counts.loc[counts.index.isin(val.index), 'Dataset'] = dataset_labels['Validation']
    counts.loc[counts.index.isin(tes.index), 'Dataset'] = dataset_labels['Test']

    order = {
        dataset_labels['Validation']: 0,
        dataset_labels['Training']: 2,
        dataset_labels['Test']: 1
    }
    counts['Priority'] = counts['Dataset'].map(order)
    counts = counts.sort_values(by='Priority')
    
    metric['Dataset'] = dataset_labels['Training']
    metric.loc[metric.index.isin(val.index), 'Dataset'] = dataset_labels['Validation']
    metric.loc[metric.index.isin(tes.index), 'Dataset'] = dataset_labels['Test']
    metric['Priority'] = metric['Dataset'].map(order)
    metric = metric.sort_values(by='Priority')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    palette = {
        dataset_labels['Training']: 'royalblue',
        dataset_labels['Validation']: 'darkorange',
        dataset_labels['Test']: 'green'
    }

    sns.histplot(data=counts, x='count', hue='Dataset', bins=20, kde=False,
                 palette=palette, alpha=0.6, ax=ax1)
    ax1.set_xlabel('Number of Cells per File', fontsize=18)
    ax1.set_ylabel('Frequency', fontsize=18)
    ax1.set_title('Distribution of Cell Count', fontsize=18)
    sns.histplot(data=metric, x='SNR', hue='Dataset', bins=20, kde=False,
                 palette=palette, alpha=0.6, ax=ax2)
    ax2.set_xlabel('SNR', fontsize=18)
    ax2.set_ylabel('Frequency', fontsize=18)
    ax2.set_title('Distribution of SNR', fontsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

