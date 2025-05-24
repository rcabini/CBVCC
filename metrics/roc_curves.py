import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import re

#---------------------------------------------------------------

def clean_model_name(name):
    name = re.sub(r'\s*\(.*?\)', '', name).strip()
    name = name.replace('University of Washington', 'UWT-SET')
    return name

#---------------------------------------------------------------

def plot_roc_curves(all_data, output_path='./roc.png', style_dict=None):
    y_true = all_data['gt']
    linestyles = ['--', '-', '-.', ':']
    colors = ['darkturquoise', 'gold', 'orangered', 'mediumslateblue', 'darkviolet', 'yellowgreen', 'royalblue']

    roc_auc_dict = {}
    handles_dict = {}

    if style_dict is None:
        style_dict = {}

    plt.figure(figsize=(6, 5), tight_layout=True)

    for i, model_name in enumerate(all_data.columns):
        if model_name == 'gt':
            continue
        y_pred = all_data[model_name]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        roc_auc_dict[model_name] = roc_auc

        if model_name not in style_dict:
            linestyle = linestyles[i % len(linestyles)]
            color = colors[i % len(colors)]
            style_dict[model_name] = {'linestyle': linestyle, 'color': color}

        style = style_dict[model_name]
        line, = plt.plot(fpr, tpr, linestyle=style['linestyle'], color=style['color'], linewidth=1.5)
        handles_dict[model_name] = line
        
    random_line, = plt.plot([0, 1], [0, 1], '--', color='lightgray')
    sorted_models = sorted(roc_auc_dict.items(), key=lambda x: x[1], reverse=True)
    handles = [handles_dict[model_name] for model_name, _ in sorted_models]
    handles.append(random_line)

    labels = [
        f'{clean_model_name(model_name)} (AUC = {auc_value:.3f})'
        for model_name, auc_value in sorted_models
    ]
    labels.append('Random Classifier')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(handles=handles, labels=labels, loc='lower right')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return style_dict
