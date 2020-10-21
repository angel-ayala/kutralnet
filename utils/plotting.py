#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:53:26 2020

@author: Angel Ayala <angel4ayala [at] gmail.com>
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from models import get_model_paths

pio.templates['kutralnet'] = go.layout.Template(
    layout_colorway=px.colors.qualitative.T10)
pio.templates.default = "plotly_white+kutralnet"


def get_results_path(models_root, create_path=False):
    """Get (and create) results path for graphs storage."""
    # main folder
    models_result_path = os.path.join(models_root, 'results')
    # if must create
    if create_path:
        # models root
        if not os.path.exists(models_root):
            os.makedirs(models_root)
        # model result path folder
        if not os.path.exists(models_result_path):
            os.makedirs(models_result_path)
    
    else:
        # dir exists?
        if not os.path.isdir(models_result_path):
            models_result_path = None
            
    return models_result_path


def get_data(models_root, model_id, dataset_id, dataset_test_id=None, version=None):
    """Get the training and testing results of a model."""
    save_path, _ = get_model_paths(models_root, model_id, 
                                                 dataset_id, version)
    # check if was trained
    if save_path is None:
        return None, None, None
    
    # training summary
    try:
        training_data = pd.read_csv(os.path.join(save_path, 'training_summary.csv'),
                                header=None)
    except:
        print('No training file for {} trained with {} ({})'.format(
                model_id, dataset_id, version))
        training_data = list()
        
    # testing summary
    try:
        testing_filename = 'testing_summary.csv' if (
            dataset_test_id is None) else 'testing_summary_{}.csv'.format(dataset_test_id)
        testing_data = pd.read_csv(os.path.join(save_path, testing_filename),
                                   header=None)
        # metrics
        metrics_filename = 'testing_metrics.pkl' if (
            dataset_test_id is None) else 'testing_metrics_{}.pkl'.format(dataset_test_id)
                    
        metrics = list()
        with open(os.path.join(save_path, metrics_filename), 'rb') as f:
            metrics.append(pickle.load(f)) # reports
            metrics.append(pickle.load(f)) # matrices
            metrics.append(pickle.load(f)) # roc_data        
            
    except:
        print('No {} test file for {} trained with {} ({})'.format(
                dataset_test_id, model_id, dataset_id, version))
        testing_data = list()
        metrics = list()
    
    return training_data, testing_data, metrics


def process_metrics(metrics):
    reports, _, roc_data = metrics  # skip matrices
    metrics_data = dict()
    
    # separated labels reports
    report = reports[0]  
    if 'accuracy' in report.keys() and len(reports) == 1:
        metrics_data['accuracy_test'] = report['accuracy']
    metrics_data['precision_avg_weight'] = report['weighted avg'
                                                   ]['precision']
    metrics_data['precision_avg_macro'] = report['macro avg'
                                                  ]['precision']

    for auroc in roc_data:
        metrics_data[auroc['label']+'_auroc'] = auroc['auroc']
        metrics_data[auroc['label']+'_precision'] = report[
                                        auroc['label']]['precision']
        
    if len(reports) == 2:  # fire and smoke reports
        report = reports[1]
        if 'accuracy' in report.keys():
            metrics_data['eme_accuracy_test'] = report['accuracy']
            
        metrics_data['eme_precision_avg_weight'] = report['weighted avg'
                                                          ]['precision']
        metrics_data['eme_precision_avg_macro'] = report['macro avg'
                                                         ]['precision']
        metrics_data['Emergency_precision'] = report['Emergency']['precision']
        
    return metrics_data


def get_roc_values(roc_metrics):
    metric_df = None
    # process metrics by label
    for roc in roc_metrics:
        label_metrics = dict()
        label_metrics['label'] = roc['label']
        label_metrics['auroc'] = roc['auroc']
        label_metrics['FPR'] = roc['roc']['FPR']
        label_metrics['TPR'] = roc['roc']['TPR']
        label_metrics['ACC'] = roc['acc']['ACC']
        label_df = pd.DataFrame.from_dict([label_metrics])
        metric_df = pd.concat([metric_df, label_df])

    return metric_df


def get_model_info(training_data):
    """Get the model stats from summary."""
    model_flops = training_data.loc[training_data[0] == 'Model FLOPS'].iat[0, 1]
    model_params = training_data.loc[training_data[0] == 'Model parameters'].iat[0, 1] 
    return model_flops, model_params


def get_training_info(training_data):
    """Get the model and dataset names from summary."""
    dataset_name = training_data.loc[training_data[0] == 'Training dataset'].iat[0, 1]
    model_name = training_data.loc[training_data[0] == 'Model name'].iat[0, 1]
    return model_name, dataset_name


def get_testing_info(testing_data):
    """Get the model and dataset names from summary."""
    test_acc = testing_data.loc[
        testing_data[0] == 'Testing accuracy'].iat[0, 1]
    test_time = testing_data.loc[
        testing_data[0] == 'Testing time (s)'].iat[0, 1]
    return test_acc, test_time


def get_validation_acc(training_data):
    """Get the validation data from summary."""
    val_acc = training_data.loc[training_data[0] == 'Validation accuracy'].iat[0, 1]
    best_epoch = training_data.loc[training_data[0] == 'Best ep'].iat[0, 1]
    return float(val_acc), int(best_epoch)
    

def get_testing_acc(testing_data):
    """Get the testing data from summary."""    
    test_acc = testing_data.loc[testing_data[0] == 'Testing accuracy'].iat[0, 1]
    auroc_val = testing_data.loc[testing_data[0] == 'AUROC value'].iat[0, 1]
    
    if '[' in auroc_val:
        auroc_val = ast.literal_eval(auroc_val)
    else:
        auroc_val = float(auroc_val)
    
    return float(test_acc), auroc_val


def summary_csv(models_root, models, datasets, test_datasets, filename=None, 
                versions=None):
    """Generate a summary CSV from training and test files."""
    must_save = not filename is None
    results_path = get_results_path(models_root, create_path=must_save)

    csv_data = None
    csv_metrics = None

    if versions is None:
        versions_list = [None]

    elif isinstance(versions, list):
        versions_list = versions

    if test_datasets is None:
        test_datasets = [None]

    for model_id in models:
        for dataset_id in datasets:        
            if isinstance(versions, str) and versions == 'all':
                model_path, _ = get_model_paths(models_root, model_id, 
                                                dataset_id)
                versions_list = os.listdir(model_path)

            for dataset_test_id in test_datasets: 
                for version_id in versions_list:
                    summary_data = get_data(models_root, 
                                            model_id, dataset_id, 
                                            dataset_test_id=dataset_test_id,
                                            version=version_id)
                    
                    if summary_data[0] is None:
                        continue
                    
                    model_name, dataset_name = get_training_info(summary_data[0])
                    model_flops, model_params = get_model_info(summary_data[0])
                    
                    if len(summary_data[0]) == 0:  # training_data
                        val_acc, best_epoch = float('NaN'), float('NaN')
                    else:
                        val_acc, best_epoch = get_validation_acc(
                                                            summary_data[0])
                    
                    if len(summary_data[1]) == 0:  # testing_data
                        test_acc, test_time = float('NaN'), float('NaN')
                        metrics = dict()
                    else:
                        test_acc, test_time = get_testing_info(summary_data[1])                        
                        metrics = process_metrics(summary_data[2])
                        roc_metrics = get_roc_values(summary_data[2][2])
                   
                    common_data = dict(model_id= model_id,
                        model_name= model_name,
                        dataset_id= dataset_id,
                        dataset_name= dataset_name,
                        dataset_test_id= dataset_test_id,
                        version_id= version_id)
                    
                    # metrics                    
                    metrics_df = pd.DataFrame.from_dict([common_data])
                    metrics_df = metrics_df.join(roc_metrics)
                    csv_metrics = pd.concat([csv_metrics, metrics_df])
                    
                    # summary
                    summ_data = dict(best_epoch= best_epoch,
                        val_acc= val_acc,
                        test_acc= test_acc,
                        test_time= test_time,
                        model_flops= model_flops, 
                        model_params= model_params
                        )
                    
                    common_data.update(summ_data)
                    common_data.update(metrics)
                    
                    summ_df = pd.DataFrame.from_dict([common_data])                        
                    csv_data = pd.concat([csv_data, summ_df])
                    
                    
    # csv_df = pd.DataFrame.from_dict(csv_data)
    # no numeric data fix
    csv_data['test_time'] = csv_data['test_time'].astype('float32')
    csv_data['test_acc'] = csv_data['test_acc'].astype('float32')
    
    # replace for short names
    csv_data['model_name'] = csv_data['model_name'].str.replace(
        'KutralNet ', 'KN ').str.replace('Mobile ', 'MB ')
    csv_metrics['model_name'] = csv_metrics['model_name'].str.replace(
        'KutralNet ', 'KN ').str.replace('Mobile ', 'MB ')

    if must_save:
        results_path = get_results_path(models_root, create_path=must_save)
        csv_data.to_csv(os.path.join(results_path, 'results_' + filename), 
                        index=None)
        csv_metrics.to_csv(os.path.join(results_path, 'metrics_' + filename), 
                           index=None)
        
    return csv_data, csv_metrics


def plot_samples(data):
    """Plot a sample of image from data."""
    fig = plt.figure()

    for i in range(len(data)):
        sample = data[i]

        print(i, sample[0].shape, sample[1].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        label = 'Fire' if sample[1] == 1 else 'Nofire'
        ax.set_title('Sample {}'.format(label))
        ax.axis('off')
        img = sample[0].transpose(2, 0)
        plt.imshow(img.transpose(0, 1))

        if i == 3:
            plt.show()
            break
# end show_samples

def plot_history(history, folder_path=None):
    """Plot history file from training."""
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.tight_layout()

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'accuracy.png'))

    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.tight_layout()

    if folder_path is not None:
        plt.savefig(os.path.join(folder_path, 'loss.png'))

    plt.show()
# end plot_history

def plot_bars(summ_data, models, datasets, title='Test accuracy', 
              column='test_acc', yrange=[.50, 1.01], label=None,
              filename=None):
    assert column in summ_data.columns.values

    columns_index = ['model_id', 'model_name', 'dataset_id', 'dataset_name',
                     'dataset_test_id']
    if 'label' in summ_data.columns.values:
        columns_index.append('label')

    bar_data = summ_data.groupby(columns_index).agg({column: ['mean', 'std']})
    
    prefilters = np.in1d(bar_data.index.get_level_values('dataset_id'), 
                        datasets)
    
    if 'label' in columns_index:
        prefilters = np.logical_and(prefilters,
                        np.in1d(bar_data.index.get_level_values('label'), 
                                [label]))
        
    bars = []
    for m in models:
        filters = np.logical_and(prefilters,
            np.in1d(bar_data.index.get_level_values('model_id'), 
                    [m]))
        
        model_data = bar_data[filters][column]
        mean_val = model_data['mean'].to_numpy()
        std_val = model_data['std'].to_numpy()
        
        model_name = model_data.index.get_level_values('model_name').values[0]
        datasets_name = model_data.index.get_level_values('dataset_name').values

        bars.append(go.Bar(x=datasets_name, y=mean_val, name=model_name, 
                           error_y=dict(type='data', 
                                        array=std_val))
        )

    fig = go.Figure(data=bars)

    fig.update_layout(dict(
        title_text = title,
        title=dict(x=.5, y=.99),
        # titlefont=dict(size=20),
        legend=dict(x=.75, y=0, font=dict(size=14)),
        yaxis_tickformat = '.02%',
        # xaxis_title='Dataset',
        yaxis_title='Accuracy',
        yaxis_range = yrange,
        margin=dict(l=20, r=20, t=30, b=20),
        width=800, height=600
        )
    )
    
    if filename is not None:
        print('Saving at', filename)
        fig.write_image(filename)
        
    return fig


def plot_time(fire_summary, models, datasets, filename=None):
    fig = plot_bars(fire_summary, models, datasets,
          title='Test time', column='test_time')
    
    fig.update_layout(dict(
            yaxis_tickformat = '',
            yaxis_range=None,
            yaxis_title='Time (s)',
            legend=dict(x=.5, y=-0.25, font=dict(size=14),
                        orientation="h",
                        yanchor="bottom", xanchor="center",),
            )
        )
    
    if filename is not None:
        print('Saving at', filename)
        fig.write_image(filename)
    
    return fig


def plot_roc(metrics_data, models, dataset, label='Fire', filename=None):
    columns_index = ['model_id', 'model_name', 'dataset_id',  
                     'dataset_test_id', 'label']

    def mean_list(l):
        # return a tuple to be an object
        if '[' == l.values[0][0]:
            l = [ ast.literal_eval(v) for v in l ]
        return tuple(np.mean([l_i for l_i in l], axis=0))

    roc_metrics = metrics_data.groupby(columns_index).agg(
        dict(auroc= ['mean'],
             FPR= [('mean', lambda x: mean_list(x) )],
             TPR= [('mean', lambda x: mean_list(x) )]))

    prefilters = np.logical_and(
        np.in1d(roc_metrics.index.get_level_values('dataset_id'), 
                [dataset]),
        np.in1d(roc_metrics.index.get_level_values('label'), 
                [label]))
    
    lines = []
    for m in models:
        filters = np.logical_and(prefilters, 
                    np.in1d(roc_metrics.index.get_level_values('model_id'), 
                                [m]))
        line_data = roc_metrics[filters]
        auc_val = line_data['auroc'].values[0]
        # print(list(roc_metrics[filters]['FPR']['mean'].values[0]))
        model_name = line_data.index.get_level_values('model_name').values[0]
        lines.append(
            go.Scatter(x=list(line_data['FPR']['mean'].values[0]), 
                       y=list(line_data['TPR']['mean'].values[0]), 
                       name=f"{model_name} AUC={auc_val[0]:.2f}", 
                       mode='lines'))
    
    filter = metrics_data['dataset_id'] == dataset
    dataset_name = metrics_data[filter]['dataset_name'].iat[0]

    fig = go.Figure(data=lines)
    # diagonal
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_layout(dict(
        title_text = 'ROC curve in {} for {} label'.format(dataset_name, label),
        legend=dict(x=.6, y=0, font=dict(size=14)),
        yaxis_tickformat = '.1%',
        xaxis_tickformat = '.1%',
        title=dict(x=.5, y=.99),
        # titlefont=dict(size=20),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=600, height=500
        )
    )
    
    if filename is not None:
        print('Saving at', filename)
        fig.write_image(filename)
    
    return fig


if __name__ == '__main__':
    root_path = os.path.join('..')
    models_root = os.path.join(root_path, 'kutralnet_extension')
    print('Root path:', root_path)
    print('Models path:', models_root)
    
    # fire only
    filename = 'fire_experiment.csv'
    fire_models = [# state-of-the-art models
          # 'firenet', 'octfiresnet', 'resnet', 
          'efficientnet', 'fire_detection', 'resnet18',
          # proposal 
          'kutralnet', 'kutralnet_pre', 'kutralnetoct',
          'kutralnet_mobile', 'kutralnet_mobileoct', 'kutralnet_mobileoct_pre']

    fire_datasets = ['firenet_relabeled', 'fismo_relabeled', 'fismo_black_relabeled']
    fire_test_datasets = ['firenet_test_relabeled']
    
    __activations = ['ce_softmax', 'focal_sigmoid', 'cb_focal_sigmoid']
    __versions = ['run_0', 'run_1', 'run_2', 'run_3', 'run_4']
    
    versions = [ act +'/'+ ver for act in __activations for ver in __versions]
    
    summary_csv(models_root, fire_models, fire_datasets, 
                fire_test_datasets, versions=versions, filename=filename)
    
    
    results_path = get_results_path(models_root)
    print('Results path:', results_path)
    
    fire_summary = pd.read_csv(os.path.join(results_path,
                                            'results_' + filename))
    fire_metrics = pd.read_csv(os.path.join(results_path,
                                            'metrics_' + filename))
    
    # Validation accuracy plot
    plot_bars(fire_summary, fire_models, fire_datasets,
          title='Validation accuracy', column='val_acc', 
          yrange=[.40, 1.01], 
          filename=os.path.join(results_path, 'fire_validation.pdf')).show()
    
    # Test accuracy plot
    plot_bars(fire_summary, fire_models, fire_datasets,
          title='Test accuracy', column='test_acc', yrange=[.40, 1.01],
          filename=os.path.join(results_path, 'fire_test.pdf')).show()
    
    # Test time plot
    plot_time(fire_summary, fire_models, fire_datasets[0],
              filename=os.path.join(results_path, 'fire_time.pdf')).show()
    
    # ROC curve plot
    for ds in fire_datasets:
        plot_roc(fire_metrics, fire_models, dataset=ds, label='Fire',
                 filename=os.path.join(results_path, 'fire_roc_' + ds +'.pdf')
                 ).show()
        
    # AUROC values comparison
    plot_bars(fire_metrics, fire_models, fire_datasets,
              title='ROC AUC Fire label', column='auroc', 
              yrange=[.40, 1.01], label='Fire',
              filename=os.path.join(results_path, 'fire_auroc.pdf')
              ).show()
    
    
    # fire and smoke
    filename = 'fire_smoke_experiment.csv'
    models = [# state-of-the-art models
          'efficientnet', 'fire_detection','resnet18',
          # proposal 
          'kutralnet', 'kutralnet_pre', 'kutralnetoct',
          'kutralnet_mobile', 'kutralnet_mobileoct', 'kutralnet_mobileoct_pre']
    datasets = ['fismo_v2', 'fireflame_v2']
    test_datasets = ['fireflame_testv2']
    
    __activations = ['bce_sigmoid', 'focal_sigmoid', 'cb_focal_sigmoid']
    __versions = ['run_0', 'run_1', 'run_2', 'run_3', 'run_4']
    
    versions = [ act +'/'+ ver for act in __activations for ver in __versions]

    summary_csv(models_root, models, datasets, 
                test_datasets, versions=versions, filename=filename)
    
    summary = pd.read_csv(os.path.join(results_path,
                                            'results_' + filename))
    metrics = pd.read_csv(os.path.join(results_path,
                                            'metrics_' + filename))
    
    # Validation accuracy plot
    plot_bars(summary, models, datasets,
          title='Validation accuracy', column='val_acc', 
          yrange=[.40, 1.01], 
          filename=os.path.join(results_path, 'fire_smoke_validation.pdf')
          ).show()
    
    # Test accuracy plot
    plot_bars(summary, models, datasets,
          title='Test accuracy', column='test_acc', yrange=[.40, 1.01],
          filename=os.path.join(results_path, 'fire_smoke_test.pdf')).show()
    
    # Test time plot
    plot_time(summary, models, datasets[0],
              filename=os.path.join(results_path, 'fire_smoke_time.pdf')).show()
    
    # ROC curve plot
    for ds in datasets:
        plot_roc(metrics, models, dataset=ds, label='Fire',
                 filename=os.path.join(results_path, 'fire_smoke_roc_' + ds +'_fire.pdf')
                 ).show()
        plot_roc(metrics, models, dataset=ds, label='Smoke',
                 filename=os.path.join(results_path, 'fire_smoke_roc_' + ds +'_smoke.pdf')
                 ).show()
        
    # AUROC values comparison
    plot_bars(metrics, models, datasets,
              title='ROC AUC Fire label', column='auroc', 
              yrange=[.40, 1.01], label='Fire',
              filename=os.path.join(results_path, 'fire_smoke_auroc_fire.pdf')
              ).show()
              
    plot_bars(metrics, models, datasets,
              title='ROC AUC Smoke label', column='auroc', 
              yrange=[.40, 1.01], label='Smoke',
              filename=os.path.join(results_path, 'fire_smoke_auroc_smoke.pdf')
              ).show()
