#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Mar 9 17:30 2020

@author: khayes847
"""

import warnings
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


def train_test(x_val, y_val, test=30000, rs_val=42):
    """
    Seperates values into train and test data.

    Input:
    x_val: Dataset features
    y_val: Dataset label
    test: Percentage of dataset will be test
    rs_val: Random seed

    Output:
    X_train: Training features
    y_train: Training labels
    X_test: Test features
    y_test: Test labels
    """
    x_train, x_test, y_train, y_test = train_test_split(x_val, y_val,
                                                        test_size=test,
                                                        random_state=rs_val,
                                                        stratify=y_val)
    return x_train, x_test, y_train, y_test


def smote_under(x_train, y_train, smote_ss = 0.25, under_ss = 0.75,
               rs_val = 42):
    """
    Creates artificial training dataset data points for "1" label,
    undersamples "0" label.

    Input:
    x_train: Training dataset features.
    y_train: Training dataset labels.
    smote_ss: Percentage of minority label in artificial dataset.
    under_ss: Percentage of majority label that will be kept in
              artificial dataset.
    rs_val: Random state value.

    Output:
    x_train: Features for artificial training dataset.
    y_train: Labels for artificial training dataset.
    """
#   Create list of column names for x_train and y_train
    x_cols = list(x_train.columns)
    y_cols = list(y_train.columns)

#   Create artificial SMOTE data points for minority label,
#   undersample majority label.
    over = SMOTE(sampling_strategy=smote_ss, random_state=42)
    under = RandomUnderSampler(sampling_strategy=0.75, random_state=42)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    x_train, y_train = pipeline.fit_resample(x_train, y_train)

#   Change new dataset into Pandas dataframes.
    x_train = pd.DataFrame(x_train, columns=x_cols)
    y_train = pd.DataFrame(y_train, columns=y_cols)
    return x_train, y_train


def cap_outliers(train, dev, test, cap=0.007):
    """
    Sets a ceiling and a floor cap for outliers
    from each feature using the training set. Caps outliers
    from training, dev, and test sets using this.

    Inputs:
    train: Training dataset.
    dev: Dev dataset.
    test: Test dataset.
    cap: Percentage used to define outliers.

    Outputs:
    train: Training dataset, capped.
    dev: Dev dataset, capped.
    test: Test dataset, capped.
    """
    for col in list(train.iloc[:,:-1].columns):
        per = train[col].quantile([cap,(1-cap)]).values
        for df in [train, dev, test]:
            df[col] = df[col].apply(lambda x: per[0] if x<per[0]
                                    else (per[1] if x>per[1]
                                          else x))
    return train, dev, test
