#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:22:19 2019

@author: hananhindy
"""
import argparse
import pandas as pd
import helper 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os 
import numpy as np
import autoencoder
import oneclass_svm

from datetime import datetime


def evaluate(model, valid_X, attack_path, output_file):
    if "OneClassSVM" in str(type(model)):
        pred = model.predict(valid_X)
        res = np.count_nonzero(np.array(pred) == 1) / np.size(pred)
        
        with open(output_file, "a") as file:
            file.write('{}, {}, {} \n'.format(attack_path, res, 1-res))
    else:
        threshold = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        res = model.evaluate(valid_X, valid_X)        
        with open(output_file, "a") as file:
            file.write('Result using threshold = 0.05 \n')
            file.write('{}, {}, {}, {} \n'.format(attack_path, res[1], 1-res[1], np.shape(valid_X)[0]))
                
        for th in threshold:
            anomaly_count = (np.mean(np.square(valid_X - model.predict(valid_X)), axis=1) < th).sum()
            res_ = anomaly_count / np.shape(valid_X)[0]
            with open(output_file, "a") as file:
                file.write('{}, {}, {}, {}, {}\n'.format(th, '', res_, 1-res_, anomaly_count))
            
            
label_encoder_1 = preprocessing.LabelEncoder()
label_encoder_2 = preprocessing.LabelEncoder()
label_encoder_3 = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder(categorical_features = [1,2,3])

def read_kdd_dataset(path):
    global label_encoder_1, label_encoder_2, label_encoder_3, one_hot_encoder
    
    dataset = pd.read_csv(path, header=None)
    if '+' not in path:
        dataset[41] = dataset[41].str[:-1]

    dataset[42] = ''
    dataset = dataset.values
    dataset = helper.add_kdd_main_classes(dataset)
    
    if hasattr(label_encoder_1, 'classes_') == False:
        dataset[:, 1] = label_encoder_1.fit_transform(dataset[:, 1])
        dataset[:, 2] = label_encoder_2.fit_transform(dataset[:, 2])        
        dataset[:, 3] = label_encoder_3.fit_transform(dataset[:, 3])
        dataset_features = one_hot_encoder.fit_transform(dataset[:, :-2]).toarray() 
    else:
        dataset[:, 1] = label_encoder_1.transform(dataset[:, 1])
        dataset[:, 2] = label_encoder_2.transform(dataset[:, 2])        
        dataset[:, 3] = label_encoder_3.transform(dataset[:, 3])
        dataset_features = one_hot_encoder.transform(dataset[:, :-2]).toarray() 
        
    return dataset, dataset_features
    
if __name__ == '__main__':
   
    columns_to_drop = ['ip_src', 'ip_dst']

    parser = argparse.ArgumentParser()
    parser.add_argument('--normal_path', default='DataFiles/CIC/biflow_Monday-WorkingHours_Fixed.csv')
    parser.add_argument('--attack_paths', default='DataFiles/CIC/')
 #   parser.add_argument('--dataset_path', default='DataFiles/KDD/kddcup.data_10_percent_corrected')
#    parser.add_argument('--dataset_path', default='DataFiles/NSL/KDDTrain+.txt')
    
    parser.add_argument('--output', default='Results.csv')
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--archi', default='U15,U9,U15')
#    parser.add_argument('--archi', default='U100,D,U60,D,U100')
    parser.add_argument('--regu', default='l2')
    parser.add_argument('--l1_value',type=float, default=0.01)
    parser.add_argument('--l2_value',type=float, default=0.0001)
    parser.add_argument('--correlation_value',type=float, default=0.9)
    parser.add_argument('--dropout',type=float, default=0.05)
    parser.add_argument('--model', default = 'autoencoder')
    parser.add_argument('--nu',type=float, default=0.01)
    parser.add_argument('--kern', default='rbf')
    parser.add_argument('--loss', default='mse')
    
    
    args = parser.parse_args()
    output_file = args.output;

    output_file = datetime.now().strftime("%d_%m_%Y__%H_%M_") + output_file
    helper.file_write_args(args, output_file)    
    
    if('dataset_path' in args):
        dataset, dataset_features = read_kdd_dataset(args.dataset_path)
         
        normal = dataset_features[dataset[:, 42] == 'normal', :]
        
        standard_scaler = preprocessing.StandardScaler()
        normal = pd.DataFrame(standard_scaler.fit_transform(normal))
        
        if 'Train+' in args.dataset_path:
            dataset_testing, dataset_features_testing = read_kdd_dataset(args.dataset_path.replace('Train+', 'Test+'))
            dataset_features_testing = standard_scaler.transform(dataset_features_testing)
    else:
        normal = pd.read_csv(args.normal_path)
        normal = normal.dropna()
        normal.drop(columns_to_drop, axis=1, inplace=True)
        normal.drop(normal.columns[0], axis=1, inplace=True)
        normal, to_drop = helper.dataframe_drop_correlated_columns(normal, args.correlation_value)
        
        standard_scaler = preprocessing.StandardScaler()
        x_scaled = standard_scaler.fit_transform(normal.values)
        normal = pd.DataFrame(x_scaled)
        print(normal.columns)
        
    train_X, valid_X, train_ground, valid_ground = train_test_split(
                                                            normal,
                                                            normal, 
                                                            test_size=0.25, 
                                                            random_state=1)
    
    if args.model == 'svm':
        wrapper = oneclass_svm.oneclass_svm(args.nu, args.kern)
        wrapper.model.fit(train_X)

    elif args.model == 'autoencoder':    
        wrapper = autoencoder.autoencoder(len(normal.columns), 
                                     archi = args.archi, 
                                     reg = args.regu,
                                     l1_value = args.l1_value, 
                                     l2_value = args.l2_value,
                                     dropout = args.dropout,
                                     loss = args.loss)
        
        with open(output_file ,'a') as file:
            wrapper.model.summary(print_fn=lambda x: file.write(x + '\n'))    
        
        hist = wrapper.model.fit(train_X, 
                 train_ground,
                 batch_size=1024,
                 epochs=args.epochs, 
                 validation_data=(valid_X, valid_ground))
    
        helper.plot_model_history(hist,'{}'.format(output_file.replace('.csv', '.pdf')))
        
        res1 = wrapper.model.evaluate(train_X, train_ground)   
        res2 = wrapper.model.evaluate(valid_X, valid_ground)

        with open(output_file, "a") as file:
            file.write('Training accuracy, {} \n Validation accuracy,{}\n'.format(res1[1], res2[1]))

    else:
        raise Exception('You should specify the model single class svm or autoencoder' )
        
    
    with open(output_file, "a") as file:
        file.write('File, accuracy, detection accuracy (1- AE acc]\n')

    evaluate(wrapper.model, train_X, "training", output_file)
    evaluate(wrapper.model, valid_X, "validation", output_file)
    if('dataset_path' in args):
        for a in ['dos', 'r2l', 'u2r', 'probe']:
            attack = dataset_features[dataset[:, 42] == a, :]
            attack = pd.DataFrame(standard_scaler.transform(attack))
            evaluate(wrapper.model, attack, a, output_file)
        
        if 'Train+' in args.dataset_path:
            for a in ['dos', 'r2l', 'u2r', 'probe', 'normal']:
                attack = dataset_features_testing[dataset_testing[:, 42] == a, :]
                evaluate(wrapper.model, attack, 'NSL_Test_{}'.format(a), output_file)
    else:
        for attack_path in os.listdir(args.attack_paths):
            path = os.path.join(args.attack_paths,attack_path)
            
            attack = pd.read_csv(path)
            attack = attack.dropna()
    
            attack.drop(columns_to_drop, axis=1, inplace=True)
            attack.drop(attack.columns[0], axis=1, inplace=True)
            attack.drop(to_drop, axis=1, inplace=True)
            
            x_scaled = standard_scaler.transform(attack.values)
            attack = pd.DataFrame(x_scaled)
            
            evaluate(wrapper.model, attack, attack_path, output_file)
