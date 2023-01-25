#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:45:25 2022

@author: gilson
"""

import pickle
import numpy as np
import pandas as pd


X = [x for x in range(-5, 5)]

Y = [x**2 for x in X]

tuple_x_y = list(zip(X, Y))

df = pd.DataFrame(tuple_x_y, columns=['x_train','y_train'])

path_name = r"data_dir/locdata.pkl"

df.to_pickle(path_name)

def data_func(data_arguments):
    """Define data function that loads pickled data from filename and yield mini-batch of batch_size """
    # 1st element from data_arguments is filename
    # 2nd element is batch size
    filename, batch_size = data_arguments

    # load pickled data from filename
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    x_train = data['x_train']
    y_train = data['y_train']

    # calculate number of mini-batches, suppose 1st dimension of x_train is #samples
    N = x_train.shape[0]
    steps = int(np.ceil( N / float(batch_size)))

    # give definition of generator

    for step in range(steps):
        start_idx = step*batch_size
        stop_idx = min(N, (step+1)*batch_size)

        yield x_train[start_idx:stop_idx], y_train[start_idx:stop_idx]

def data_func_train(data_arguments):
    """Define data function that loads pickled data from filename and yield mini-batch of batch_size """
    # 1st element from data_arguments is filename
    # 2nd element is batch size
    filename, batch_size = data_arguments

    # load pickled data from filename
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    x_train = data['x_train']

    # calculate number of mini-batches, suppose 1st dimension of x_train is #samples
    N = x_train.shape[0]
    steps = int(np.ceil( N / float(batch_size)))

    # give definition of generator

    for step in range(steps):
        start_idx = step*batch_size
        stop_idx = min(N, (step+1)*batch_size)

        yield x_train[start_idx:stop_idx]

data_arguments = (path_name, 3)

data_gen = data_func(data_arguments)

data_gen_train = data_func_train(data_arguments)

print(next(data_gen))

print(next(data_gen_train))

# Now define data_argument