# Neural-network

This is a simple artificial neural network in Python using Numpy. 
The network is using gradient descent for optimisation and sigmoid function is used for activation.

## Usage

First, clone the repository with `git clone`. 

### Training and testing data

To test out the network, create some demo data with `test_data_generator.py`. This will produce two text files with input and output data for training. Generated data constitutes a simple classification problem of two groups of vectors at different distances from the origin. In a similar fashion you can generate data for testing, however the files will need to have distinct names.

### Setting up the network

To construct an arbitrary neural network `setup.py` will have to be modified. Choose the desired number of layers `N_l` and define the number of neurons per layer with variables `l<n>N` where `n` is the n-th layer. When defined, add all of them to the `layers` array. 

The same goes for the weights, which are randomly generated. You will have to define `N_l - 1` weights matrices, where `N_l` is the number of layers, named `w<n>` where `n` is the n-th weights matrix. When defined, add all of them to the `weights` array. 

Learning rate for gradient descent and the number of epochs can also be adjusted.

### Training the network

First the training data has to be loaded.

```
input_vectors,correct_outputs = training_data(<train_input_data_path>,<train_output_data_path>,neurons in input layer,neurons in output layer)
```
Now the neural network can be trained.

```
model = train_neural_network(input_vectors,correct_outputs,weights)
```
Training the neural network return a set of trained weights, which we call a 'model'. This model can be saved to the `models` directory using the `save_model` method and reloaded when required utilising the `load_model` method. 

### Testing the network

Testing data can be also loaded using the `training_data` method. To test the network, simply run:

```
avg_dist,dists = test_neural_network(input_vectors,correct_outputs,model)
```

This will return the average euclidian distance of the 'expected' result, produced by the model, and the correct output. It also return the entire array of distances for each test case.
