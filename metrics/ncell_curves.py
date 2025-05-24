import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, balanced_accuracy_score
)

#---------------------------------------------------------------

def compute_score_per_cell_count(all_data, counts):
    all_datac = all_data.merge(counts, left_on='file_id', right_index=True, how='left')
    metrics_dict = {'Num_Cells': [], 'Model': [], 'Score': []}

    for column in all_datac.columns:
        if column not in ['gt', 'count', 'file_id']:
            for num_cells in sorted(all_datac['count'].unique()):
                subset = all_datac[all_datac['count'] == num_cells]
                if len(subset) > 1 and 0 < num_cells <= 7:
                    predictions = (subset[column] >= 0.5).astype(int)

                    auci = roc_auc_score(subset['gt'], subset[column])
                    precision = precision_score(subset['gt'], predictions, zero_division=0)
                    recall = recall_score(subset['gt'], predictions)
                    balanced_acc = balanced_accuracy_score(subset['gt'], predictions)
                    score = round(0.4 * auci + 0.2 * (precision + recall + balanced_acc), 3)

                    metrics_dict['Num_Cells'].append(num_cells)
                    metrics_dict['Model'].append(column)
                    metrics_dict['Score'].append(score)

    return pd.DataFrame(metrics_dict)

#---------------------------------------------------------------

def plot_score_vs_cells(all_data, counts, style_dict, clean_name_fn, output_path='test_ncell_with_rank.png'):
    metrics_df = compute_score_per_cell_count(all_data, counts)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5),
                                   gridspec_kw={'height_ratios': [2, 0.2]}, sharex=True)
    model_order = []

    for model_name in metrics_df['Model'].unique():
        subset = metrics_df[metrics_df['Model'] == model_name]
        truncated_name = clean_name_fn(model_name)
        style = style_dict.get(model_name, {'linestyle': '-', 'color': 'black'})
        
        ax1.plot(
            subset['Num_Cells'] - 1,
            subset['Score'],
            label=truncated_name,
            linestyle=style['linestyle'],
            color=style['color'],
            marker='o',
            linewidth=1.5,
            markersize=5
        )
        model_order.append(model_name)

    ax1.set_ylabel('Score')
    ax1.set_title('Score as a function of the number of cells')
    ax1.legend()
    ax1.grid(True)

    average_scores = metrics_df.groupby(['Num_Cells', 'Model'])['Score'].mean().reset_index()
    average_scores['Rank'] = average_scores.groupby('Num_Cells')['Score'].rank(ascending=True, method='min')

    color_map = {model: style_dict.get(model, {'color': 'black'})['color'] for model in model_order}
    average_scores['Model'] = pd.Categorical(average_scores['Model'], categories=model_order, ordered=True)

    sns.barplot(
        x='Num_Cells',
        y='Rank',
        hue='Model',
        data=average_scores,
        ax=ax2,
        palette=color_map,
        hue_order=model_order
    )

    ax2.set_xlabel('Number of cells in the video-patches')
    ax2.set_ylabel('Rank')
    ax2.yaxis.set_ticks([])
    ax2.legend().set_visible(False)

    plt.setp(ax2.get_xticklabels(), rotation=0)
    plt.tight_layout(pad=-0.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

