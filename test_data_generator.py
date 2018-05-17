# -*- coding: utf-8 -*-

import numpy as np

################################################################
test_batch_size = 10000

dist1 = 1.0
dist2 = 5.0

err = 0.5

dim_input = (20,1)
dim_output = (2,1)

outputs = np.array([[[1],[0]],[[0],[1]]])
################################################################

test_batch_inputs = []
test_batch_outputs = []

while len(test_batch_inputs) < test_batch_size:
    vect = np.random.random(dim_input)
            
    dist = sum(vect ** 2)
        
    if abs(dist - dist1) < err:
        test_batch_inputs.append(vect)
        test_batch_outputs.append(outputs[0])
    elif abs(dist - dist2) < err:
        test_batch_inputs.append(vect)
        test_batch_outputs.append(outputs[1])
        
np.savetxt("data/001_input.txt",test_batch_inputs)
np.savetxt("data/001_output.txt",test_batch_outputs)
            
        
        
        
        
        
    