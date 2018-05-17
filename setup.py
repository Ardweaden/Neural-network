# -*- coding: utf-8 -*-

import numpy as np

######################## SETUP ########################

# Number of layers, including input and output layers

N_l = 3
# Number of neurons per layer

l1N = 20
l2N = 100
l3N = 2

layers = [l1N,l2N,l3N]

# Initialise weights

w1 = 2*np.random.random((l2N,l1N)) - 1
w2 = 2*np.random.random((l3N,l2N)) - 1

weights = [w1,w2]

# Learning rate for gradient descent

lr = 0.1

# Number of epochs

N_e = 1

#######################################################