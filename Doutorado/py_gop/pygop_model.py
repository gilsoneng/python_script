# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:50:13 2023

@author: Gilson.Costa
"""

from numpy.random import rand
from numpy.random import randn
import numpy as np
import pandas as pd
import pickle as pkl

def funcion_generate(X):
    Y = 100 * np.sin(X) + X ** 2
    return Y

def rand_function_escala(n, range_min = -50, range_max = 50, xrand = np.NaN):
    diff = range_max - range_min
    offset = (diff)/2
    xrand = diff * 0.05 if xrand == np.NaN else 0
    return rand(int(n))*diff - offset + randn(n)*xrand

n = 10000

X = rand_function_escala(n)

Y = funcion_generate(X)

X = X.tolist()
Y = Y.tolist()

X = [-2, -1, 0, 1, 2]

Y = [-2, -1, 0, 1, 2]

df = pd.DataFrame(X, columns=['x_train'])
df['y_train'] = pd.DataFrame(Y)

df.to_pickle(r'data_dir/data.pickle')

save = True
load = True

filename = 'data_dir/data.pickle'
fileObject = open(filename, 'wb')

if save:
    pkl.dump(df.to_numpy().T, fileObject)
    fileObject.close()

from GOP import models

model = models.POP()

params = model.get_default_parameters()

import pickle

import numpy as np

import os
import sys
# check and add current path
cwd = os.path.dirname(os.path.dirname(os.path.abspath('data_dir')))

if cwd not in sys.path:
    sys.path.append(cwd)

def data_func(data_arguments):
    """Define data function that loads pickled data from filename
       and yield mini-batch of batch_size
    """
    # 1st element from data_arguments is filename
    # 2nd element is batch size
    filename, batch_size = data_arguments
    
    # load pickled data from filename
    with open(filename, 'rb') as fid:
        data = pickle.load(fid)
    
    #print(data)
    
    x_train = data[:1]#data['x_train']
    y_train = data[1:]#data['y_train']
    
    # calculate number of mini-batches, suppose 1st dimension of x_train is #samples
    N = x_train.shape[0]
    steps = int(np.ceil( N / float(batch_size)))
    
    # give definition of generator
    
    def gen():
        while True:
            for step in range(steps):
                start_idx = step*batch_size
                stop_idx = min(N, (step+1)*batch_size)
                
                yield x_train[start_idx:stop_idx], y_train[start_idx:stop_idx]

    
    return gen(), steps # note that gen() but not gen

# Now define data_argument
data_arguments = ['data_dir/data.pickle', 128]

data_gen, steps = data_func(data_arguments)

params['tmp_dir'] = 'tmp_dir'
params['model_name'] = 'M1'
params['input_dim'] = 5
params['output_dim'] = 5
params['convergence_measure'] = 'mean_squared_error'

performance, progressive_history, finetune_history = model.fit(params, data_func, data_arguments)

