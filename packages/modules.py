import copy
from turtle import shape
import pandas as pd
import numpy as np

def train_test_split(data):
    train = data.iloc[:int(len(data)*0.8)]
    test = data.iloc[int(len(data)*0.8):]
    
    #Train
    Y_train = train["diagnosis"]
    Y_train = np.array(Y_train)
    Y_train = Y_train.reshape(Y_train.shape[0],-1).T

    X_train = train.loc[:, train.columns != "diagnosis"]
    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1).T
    
    #Test
    X_test = test.loc[:, test.columns != "diagnosis"]
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1).T

    Y_test = test["diagnosis"]
    Y_test = np.array(Y_test)
    Y_test = Y_test.reshape(Y_test.shape[0], -1).T
    
    return X_train, Y_train , X_test, Y_test

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """

    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    cache = Z 

    return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1 - s)

    return dZ

def initialize_parameters_deep(layer_dims):

    parameters = {}

    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    L = len(parameters) // 2
    A = X

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = -1/m * (np.dot(Y, np.log(AL.T)) + np.dot((1-Y), np.log(1-AL.T)))
    cost = np.squeeze(cost)
    return cost

def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * (np.dot(dZ, A_prev.T))
    db = 1/m * (np.sum(dZ, axis=1, keepdims=True))
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # initialize the backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW"+ str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+ str(l+1)], current_cache, "relu")
        grads["dA"+ str(l)] = dA_prev_temp
        grads["dW"+ str(l+1)] = dW_temp
        grads["db"+ str(l+1)] = db_temp

        return grads
    
def update_parameters(params, grads, learning_rate):
    paramters = params.copy()
    L = len(paramters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        paramters["W"+ str(l+1)] = params["W"+ str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        paramters["b"+ str(l+1)] = params["b"+ str(l+1)] - learning_rate*grads["db"+str(l+1)]

    return paramters