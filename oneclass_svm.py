#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:10:44 2019

@author: hananhindy
"""
from sklearn.svm import OneClassSVM

class oneclass_svm:
    def __init__(self, nu_value, kernel = 'rbf', verbose=True):
        self.model = OneClassSVM(nu=nu_value, kernel= kernel, gamma = 'scale', verbose=verbose)            