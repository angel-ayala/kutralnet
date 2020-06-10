import os
# file helpers
import pickle
import pandas as pd
# plot helpers
import plotly.express as px
import matplotlib.pyplot as plt
# models data
from utils.models import models_conf
from utils.training import get_paths
from datasets import available_datasets

# read models direclty from the repository's folder
root_path = os.path.join('.')
models_root, models_save_path, models_results_path = get_paths(root_path)

# for representate in graphs
ids = []
names = []

for dt in available_datasets:
    ids.append(dt)
    names.append(available_datasets[dt]['name'])

dt_names = dict(zip(ids, names))

ids = []
names = []

for m in models_conf:
    ids.append(m)
    names.append(models_conf[m]['model_name'])

models_names = dict(zip(ids, names))
models_names['firenet_tf'] = 'FireNet'
del(ids)
del(names)

def info_replace(df):
    df['base_model'] = df.base_model.replace(models_names)
    df['dataset'] = df.dataset.replace(dt_names)
    return df

# bar graphs
def plot_bar(dataframe, x_column, y_column, title=None, y_range=[.75, 1.005],
             pdf_path=None, legend_x=.75, y_title=None):
    fig = px.bar(dataframe, x=x_column, y=y_column, color='base_model', barmode='group',
             width=600, height=400, color_discrete_sequence=px.colors.qualitative.T10,
             title=title)
    fig.update_layout(dict(
        legend=dict(x=legend_x, y=0, font=dict(size=14)),
        yaxis_tickformat = '%',
        title=dict(x=.5, y=.99),
        titlefont=dict(size=20),
        margin=dict(l=20, r=20, t=30, b=20),
        legend_title_text=None,
        xaxis_title="Dataset",
        yaxis_title="{}Accuracy".format(y_title),
        )
    )
    fig.update_yaxes(range=y_range)
    fig.for_each_trace(lambda t: t.update(name=t.name.replace("base_model=","")))

    if pdf_path is not None:
        print('Saving at', pdf_path)
        fig.write_image(pdf_path)

    fig.show()

# ROC curve graphs
def plot_roc(roc_summary, img_path=None, title=False):
    lw = 2
    colors = ['darkorange', 'darkgreen', 'darkred', 'gray']

    # to get first the datasets contained
    idx = next(iter(roc_summary))
    for dt in roc_summary[idx]:
        plt.figure()
        for j, base_model in enumerate(roc_summary):
            fpr = roc_summary[base_model][dt]['fpr']
            tpr = roc_summary[base_model][dt]['tpr']
            roc_auc = roc_summary[base_model][dt]['roc_auc']
            plt.plot(fpr, tpr, color=colors[j],
                    lw=lw, label='{} AUROC={:.2f}'.format(
                        models_names[base_model], roc_auc),
                    linestyle='-.')

        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid()
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        if title:
            plt.title('ROC curve for {} dataset'.format(
                available_datasets[dt]['name']), fontsize=18)
        plt.legend(loc="lower right", prop={'size': 14})

        if img_path is not None:
            img_save_path = '{}_{}_test_results.pdf'.format(img_path, dt)
            print('Saved at', img_save_path)
            plt.savefig(img_save_path)

        plt.show()

if __name__ == '__main__':
    # baseline models plots
    training_summary_df = pd.read_csv(os.path.join(models_results_path,
                                'baseline_training_summary.csv'), index_col=False)
    testing_summary_df = pd.read_csv(os.path.join(models_results_path,
                                'baseline_test_summary.csv'), index_col=False)
    with open(os.path.join(models_results_path,
                                        'baseline_roc_summary.pkl'), 'rb') as f:
        roc_summary_dict = pickle.load(f)
    
    # training results
    plot_bar(dataframe=training_summary_df, x_column='dataset', y_column='val_acc',
             y_range=[.3, 1.005], #title='Validation accuracy results',
             y_title="Validation ", pdf_path=os.path.join(models_results_path,
                                                'baseline_training_results.pdf'))
    # testing results
    df = testing_summary_df.drop_duplicates(['dataset', 'base_model', 'accuracy'])
    plot_bar(dataframe=df, x_column='dataset', y_column='accuracy',
             y_range=[.3, 1.005], #title='Test accuracy results',
             y_title="Test ", pdf_path=os.path.join(models_results_path,
                                                    'baseline_test_results.pdf'))
    
    plot_roc(roc_summary_dict,
             img_path=os.path.join(models_results_path, 'baseline'))
    
    
    # portable models plots
    training_summary_df_p = pd.read_csv(os.path.join(models_results_path,
                                'portable_training_summary.csv'), index_col=False)
    testing_summary_df_p = pd.read_csv(os.path.join(models_results_path,
                                'portable_test_summary.csv'), index_col=False)
    with open(os.path.join(models_results_path,
                                        'portable_roc_summary.pkl'), 'rb') as f:
        roc_summary_dict_p = pickle.load(f)
    
    # training results
    plot_bar(dataframe=training_summary_df_p, x_column='dataset',
            y_column='val_acc', y_range=[.55, 1.005], legend_x=.6,
            #title='Validation accuracy results',
            pdf_path=os.path.join(models_results_path,
                                  'portable_training_results.pdf'))
    # testing results
    df = testing_summary_df_p.drop_duplicates(['dataset', 'base_model', 'accuracy'])
    plot_bar(dataframe=df, x_column='dataset', y_column='accuracy',
        y_range=[.55, 1.005], legend_x=.6, #title='Test accuracy results',
        pdf_path=os.path.join(models_results_path, 'portable_test_results.pdf'))
    
    plot_roc(roc_summary_dict_p,
             img_path=os.path.join(models_results_path, 'portable'))
