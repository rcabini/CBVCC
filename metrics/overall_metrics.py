import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    balanced_accuracy_score, 
    confusion_matrix
)

#---------------------------------------------------------------

def evaluate_models(all_data, threshold=0.5):
    metrics_dict = {
        'Model': [],
        'AUC': [],
        'Precision': [],
        'Recall': [],
        'Balanced Accuracy': [],
        'Score': [],
        'True Positive': [],
        'True Negative': [],
        'False Positive': [],
        'False Negative': []
    }

    for column in all_data.columns:
        if column == 'gt':
            continue

        predictions = (all_data[column] >= threshold).astype(int)
        y_true = all_data['gt']

        auc_val = roc_auc_score(y_true, all_data[column])
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, predictions)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

        # CVBCC overall score
        score = 0.4 * auc_val + 0.2 * (precision + recall + balanced_acc)

        metrics_dict['Model'].append(column)
        metrics_dict['AUC'].append(round(auc_val, 3))
        metrics_dict['Precision'].append(round(precision, 3))
        metrics_dict['Recall'].append(round(recall, 3))
        metrics_dict['Balanced Accuracy'].append(round(balanced_acc, 3))
        metrics_dict['Score'].append(round(score, 3))
        metrics_dict['True Positive'].append(tp)
        metrics_dict['True Negative'].append(tn)
        metrics_dict['False Positive'].append(fp)
        metrics_dict['False Negative'].append(fn)

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df = metrics_df.sort_values(by='Score', ascending=False)
    return metrics_df

