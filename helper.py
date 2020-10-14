#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:04:22 2019

@author: hananhindy
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def dataframe_drop_correlated_columns(df, threshold=0.95, verbose=False):
    if verbose:
        print('Dropping correlated columns')
    if threshold == -1:
         return df, []

    # Create correlation matrix
    corr_matrix = df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    df = df.drop(df[to_drop], axis=1)

    return df, to_drop


def file_write_args(args, file_name, one_line=False):
    args = vars(args)
    
    with open(file_name, "a") as file:
        file.write('BEGIN ARGUMENTS\n')
        if one_line:
            file.write(str(args))
        else:
            for key in args.keys():
                file.write('{}, {}\n'.format(key, args[key]))
        
        file.write('END ARGUMENTS\n')

    
def plot_probability_density(array, output_file, cutoffvalue = 2):
    array[array > cutoffvalue] = cutoffvalue
    
    # Density Plot and Histogram of all arrival delays
    plt.clf()
    
    sns_plot = sns.distplot(array, hist=True, kde=True, rug=False, fit=stats.norm,
                 color = 'darkblue', 
                 hist_kws={'edgecolor':'black'},
                 kde_kws={'linewidth': 4, 'label':'KDE'},
                 fit_kws={'color': 'red', 'linewidth': 4, 'label': 'PDF'})

    
    plt.xlabel("mse")
    plt.ylabel("Density")
    plt.title("PDF of mean square error (cut-off at mse = 2 s.th. mse > 2 is mapped to 2)") 
    plt.legend(loc='best')
    sns_plot.figure.savefig(output_file)
    
   
def plot_model_history(hist, output_file):

    plt.clf()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Accuracy', 'Validation Accuracy', 'Training Loss', 'Validation Loss'], loc='center right')
    plt.savefig(output_file)
    
def add_kdd_main_classes(dataset):
    base_classes_map = {}
    base_classes_map['normal'] =  'normal'
    base_classes_map['back'] = 'dos'
    base_classes_map['buffer_overflow'] = 'u2r'
    base_classes_map['ftp_write'] =  'r2l'
    base_classes_map['guess_passwd'] =  'r2l'
    base_classes_map['imap'] =  'r2l'
    base_classes_map['ipsweep'] =  'probe'
    base_classes_map['land'] =  'dos'
    base_classes_map['loadmodule'] =  'u2r'
    base_classes_map['multihop'] =  'r2l'
    base_classes_map['nmap'] =  'probe'
    base_classes_map['neptune'] =  'dos'
    base_classes_map['perl'] =  'u2r'
    base_classes_map['phf'] =  'r2l'
    base_classes_map['pod'] =  'dos'
    base_classes_map['portsweep'] = 'probe'
    base_classes_map['rootkit'] =  'u2r'
    base_classes_map['satan'] =  'probe'
    base_classes_map['smurf'] =  'dos'
    base_classes_map['spy'] =  'r2l'
    base_classes_map['teardrop'] =  'dos'
    base_classes_map['warezclient'] =  'r2l'
    base_classes_map['warezmaster'] =  'r2l'
    
    for key in base_classes_map:
        dataset[dataset[:, 41] == key, 42] = base_classes_map[key]
    return dataset
