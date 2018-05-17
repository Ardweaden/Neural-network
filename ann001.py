# -*- coding: utf-8 -*-

import numpy as np
import random
import os
from setup import * 

# Normalises the input vectors
def normalised(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

# Open training data, returns (inputs,outputs)
def training_data(filename_inputs,filename_outputs,n_input_layer,n_output_layer):
    inputs = np.loadtxt(filename_inputs)
    outputs = np.loadtxt(filename_outputs)
    
    inputs1 = []
    outputs1 = []
    
    for i in range(len(inputs)):
        inputs1.append(normalised(np.reshape(inputs[i],(n_input_layer,1))))
        outputs1.append(np.reshape(outputs[i],(n_output_layer,1)))
        
    return inputs1,outputs1

# Sigmoid function for activation
def sigmoid_function(x):
    try:
        result = np.exp(x)/(np.exp(x) + 1)
        return result
    except:
        result = 1/(1 + np.exp(-x))
        return result

# Error function: euclidian distance
def errfunc_euclidian_dist(output_vector,correct_output):
    return np.sqrt(sum((correct_output - output_vector)**2))

# Backpropagation
def backpropagation(output_vectors,correct_output,weights):
    for i in range(N_l - 2,0,-1):
        if i == N_l - 2:
            factor = np.dot(output_vectors[i].T,1 - output_vectors[i])
            delta = factor * (output_vectors[i] - correct_output)
            w_correction = -lr * np.outer(delta,output_vectors[i - 1])
                        
            weights[i] += w_correction
            
        else:
            factor = np.dot(output_vectors[i].T,1 - output_vectors[i])
            delta = factor * np.dot(weights[i-1].T,output_vectors[i])
            w_correction = -lr * np.outer(delta,output_vectors[i - 1])
                        
            weights[i] += w_correction
            
    return weights
    
# Forward propagation
def propagation(input_vector,weights):
    if input_vector.shape != (l1N,1):
        raise NameError("Wrong shape of the input vector. Its shape should be ({},1), but got {} instead.".format(l1N,input_vector.shape))
   
    output_vectors = []
    
    for i in range(N_l - 1):
        input_vector = np.dot(weights[i],input_vector)
        output_vectors.append(sigmoid_function(input_vector))
        
    return output_vectors

def train_neural_network(input_vectors,correct_outputs,weights):
    for j in range(len(input_vectors)):
        for i in range(N_e):
            output_vectors = propagation(input_vectors[j],weights)
            weights = backpropagation(output_vectors,correct_outputs[j],weights)
    
    return weights
            
def test_neural_network(input_vectors,correct_outputs,weights):
    dists = []
    
    for i in range(len(input_vectors)):
        _,_,dist = neural_network(input_vectors[i],weights,correct_output=correct_outputs[i])
        dists.append(dist)
        
    return sum(dists)/len(dists),dists
            
def neural_network(input_vector,weights,correct_output=[]):
    if not len(correct_output):
        return propagation(input_vector,weights)[-1]
    else:
        output = propagation(input_vector,weights)[-1]
        dist = errfunc_euclidian_dist(output,correct_output)
        return output,correct_output,dist
        
def save_model(directory,weights):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for i in range(len(weights)):
        filename = directory + "/" + str(i) + ".txt"
        np.savetxt(filename,weights[i])
        
def load_model(directory):
    weights = []    
    try:
        i = 0
        while(True):
            weights.append(np.loadtxt(directory + "/" + str(i) + ".txt"))
            i += 1
    except:
        return weights
    


