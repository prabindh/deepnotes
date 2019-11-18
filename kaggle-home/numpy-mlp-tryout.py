# Originated from https://github.com/SkalskiP/ILearnDeepLearning.py
# Changed significantly, in terms of ordering of input to match theory, 
# Added own data generator, changed matrix types, added mse, normalize, minutiae, autograd,
# Various new layers, loss types, mappings, pickled output for later reuse

import autograd.numpy as np
from autograd import grad
import math, random, time
from sklearn.model_selection import train_test_split
import sys, signal
import pickle

g_exit_signalled = 0
def signal_handler(sig, frame):
        global g_exit_signalled
        print('Setting exit flag...')
        g_exit_signalled = g_exit_signalled + 1
        if g_exit_signalled > 5:
            sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Is the dimension - that of the inputs row, or cols
# ie, if input is 1 row of 2 cols [x1, x2], is it dim 2 or 1 ?

### CONFIGURATIONS ###############################################
NN_TYPE = "regressor"  # classifier or regressor
MIN_CONTIGUOUS_HIGH_ACC_EPOCHS = 100
COST_CUTOFF = 0.001
# # number of samples in the test data set
N_SAMPLES = 100
NN_EPOCHS = 300000
# ratio between training and test sets
NN_TEST_SIZE = 0.1
NN_LEARNING_RATE = 0.205
# Debug configs
NN_DEBUG_PRINT_EPOCH_COUNT = 100
NN_DEBUG_EXIT_EPOCH_ONE = False
NN_DEBUG_SHAPES = False
NN_DEBUG_VALUES = True
NN_NORMALIZE = True
NN_SHUFFLE_ROWS_EPOCH = True
NN_BATCH_ROWS_EPOCH = False
NN_INPUT_TO_HIDDEN_MULTIPLIER = 4
NN_ZERO_MEAN_NORMALIZE = False # True will make zero mean set(with +,- values)
NN_RUN_MODE = "kaggle_home" # separated_datapoints or kaggle_home
NN_SHAPE = "long" # long, wide
###############################################################
# Global evils
g_curr_epochs = 0

if NN_TYPE == "classifier":
    NN_ARCHITECTURE_LOSS_TYPE = "cross_entropy"
elif NN_TYPE == "regressor":
    if NN_RUN_MODE == "kaggle_home":
        # "root_mean_sq_log_error" to be used, if log.Y is not taken reading from CSV. Else "root_mean_sq_error"
        NN_ARCHITECTURE_LOSS_TYPE = "root_mean_sq_log_error"  
    else:
        NN_ARCHITECTURE_LOSS_TYPE = "mean_sq_error"
def make_network_arch(input_dim, network_type=None, network_shape="long"):
    if (input_dim % 2):
        print ("WARN: Number of features not multiple, ", input_dim)
    nn = []
    final_activation = "relu"
    if network_type == "classifier":
        final_activation = "sigmoid"
    start_dim = input_dim*NN_INPUT_TO_HIDDEN_MULTIPLIER
    nn.append({"layername": "input", "input_dim": input_dim, \
            "output_dim": int(start_dim), "activation": "relu"})
    if (network_shape == "long"):
        for id in range(1000):
            if (start_dim <= 60): break
            nn.append({"layername": "hidden"+str(id+1), "input_dim": int(start_dim), \
                    "output_dim": int(start_dim/2), "activation": "relu"})
            start_dim = start_dim / 2
    elif (network_shape == "wide"):
        for id in range(3):
            nn.append({"layername": "hidden-wide"+str(id+1), "input_dim": int(start_dim), \
                    "output_dim": int(start_dim), "activation": "relu"})
    else:
        raise Exception("Unknown network_shape ", network_shape)
    nn.append({"layername": "output", "input_dim": int(start_dim), "output_dim": 1, \
            "activation": final_activation})
    return nn

def read_housing_csv(file_name, mapping_state, target_name=None):
    data = np.genfromtxt(file_name,
            delimiter=',', dtype='unicode', skip_header=1)
    if (NN_DEBUG_SHAPES):
        print (data.shape)
    if (target_name is None):
        skip_cols = 1
    else:
        skip_cols = 2
    # clean up feature-wise
    map_id = 1.0  # Dont make a feature irrelevant by making it 0
    # Map known mappings
    mapping_state["NA"] = 0.0
    mapping_state["No"] = 0.0
    mapping_state["N"] = 0.0    
    mapping_state["Unf"] = 0.0
    mapping_state["None"] = 0.0
    mapping_state["Po"] = 0.0 # Poor
    mapping_state["Y"] = map_id
    map_id = map_id + 1
    mapping_state["Fa"] = map_id
    map_id = map_id + 1
    mapping_state["TA"] = map_id
    map_id = map_id + 1
    mapping_state["Gd"] = map_id
    map_id = map_id + 1
    mapping_state["Ex"] = map_id
    map_id = map_id + 1

    # Get (samplesize x features per sample)
    X = np.empty((data.shape[0],data.shape[1]-skip_cols)) # Dont need Id and Price columns
    # Perform column-wise, so feature-wise mappings are similar, else they will be random
    for col in range(data.shape[1]-skip_cols):
        for row in range(data.shape[0]):
            try:
                X[row][col] = data[row][col+1].astype(float)
            except:
                if (data[row][col+1] in mapping_state):
                    X[row][col] = mapping_state[data[row][col+1]]
                else:
                    mapping_state[data[row][col+1]] = map_id
                    X[row][col] = map_id
                    map_id = map_id + 1.0
    # Get groundtruths
    Y = np.empty((data.shape[0],1))
    if (target_name is not None):
        for row in range(data.shape[0]):
            col = data.shape[1]-1
            try:
                # Option - Take log of saleprice to match the loss 
                Y[row][0] = data[row][col].astype(float)
            except:
                raise Exception ("Ground truth should be float")
    # Normalize
    Y_normalize_state = X_normalize_state = None
    if (NN_NORMALIZE):
        if (target_name is not None):
            Y, Y_normalize_state = normalize0(Y, NN_ZERO_MEAN_NORMALIZE, axis=0)
        X, X_normalize_state = normalize0(X, NN_ZERO_MEAN_NORMALIZE, axis=0)
    if (NN_DEBUG_SHAPES):
        print (X.shape, Y.shape, X, X[0][0].dtype)
    return X,X_normalize_state, mapping_state, Y, Y_normalize_state

def generate_separated_datapoints(angle_radians, num):
    X = []
    Labels = []
    c = 0
    x = 0
    for i in range(num):
        if i < num/2:
            c = 1
            point_class = (0,1)
        else:
            c = -1
            point_class = (1,0)
        x = i
        y = random.randint(5,20)
        
        X.append([x,y])
        # Take only class#, not one-hot encoded
        Labels.append([point_class[1]])
    X = normalize0(X)
    return np.asarray(X), np.asarray(Labels)

def print_model(nn_architecture, params):
    print ("Layer (name)\tInput_dim\tOutput_dim")    
    for idx, layer in enumerate(nn_architecture):
        print("{}\t{}\t{}".format(layer["layername"],
                    layer["input_dim"], 
                    layer["output_dim"]))

def denormalize0(data, normalize_state, zero_mean=True):
    mean = normalize_state["mean"]
    var = normalize_state["var"]
    minimum = normalize_state["min"]
    maximum = normalize_state["max"]
    if (zero_mean == True):
        denorm = np.sqrt(var+0.001) * data + mean
    else:
        denorm = data * (maximum - minimum) + minimum
    return denorm
# normalize based on each feature separately
def normalize0(data, zero_mean=True, axis=0):
    mean = np.mean(data, axis=axis)
    var = np.var(data, axis=axis)
    minimum_arr = np.amin(data, axis=axis, keepdims=True)
    maximum_arr = np.amax(data, axis=axis, keepdims=True)
    normalize_state = {"mean": mean, "var":var, "min": minimum_arr, "max": maximum_arr}
    if (zero_mean == True):
        normalized = (data - mean) / np.sqrt(var+0.001)
    else:
        normalized = (data - minimum_arr) / (maximum_arr - minimum_arr)
    return normalized.reshape(data.shape), normalize_state

def init_layers(nn_architecture, seed = 99):
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        
        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        
        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_input_size, layer_output_size) * 0.1
        params_values['b' + str(layer_idx)] = np.zeros(
            (1, layer_output_size))
        
    return params_values

# RuntimeWarning: overflow encountered in exp
# Results in sigmoid returning 0, causing issues in log later
# Need to do batchnorm
def sigmoid(Z):
    sigm = 1/(1+np.exp(-Z))
    return sigm

def relu(Z):
    return np.maximum(0,Z)

# why do we take sigmoid of the OUTPUT Z ??? YES, that is how it is backpropagated
# Why not take sigmoid of input ???
def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ
def leaky_relu(Z):
    lrelu = np.where(Z > 0, Z, Z * 0.00001)
    return lrelu
def leaky_relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0.00001 # Same coeff as forward!!
    return dZ



def evaluate_single_layer(A_prev, W_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    #  Z = A*W + b
    #    Then Activation is applied A = activate(Z), to get Input to next layer
    Z_curr = np.dot(A_prev, W_curr) + b_curr
    
    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    # return of calculated activation A and the intermediate Z matrix
    return activation_func(Z_curr), Z_curr    

def evaluate_model(X, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    A_curr = X
    
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr
        
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]
        # calculation of activation for the current layer
        A_curr, Z_curr = evaluate_single_layer(A_prev, W_curr, b_curr, activ_function_curr)
        
        # saving calculated values in the memory
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr
       
    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory

# rmsle 
def evaluate_rmsle(Y_hat, Y):
    val = np.log(Y+1.0) - np.log(Y_hat+1.0)
    cost = - np.sqrt(np.mean(val**2))
    return cost
def evaluate_rmse(Y_hat, Y):
    rmse = - np.sqrt(((Y_hat - Y) ** 2).mean())
    return rmse

# In large N (1000's), Getting RuntimeWarning: divide by zero encountered in log
# So, adding a minutiae
def evaluate_cost_value(Y_hat, Y, method, derivative=None):
    minutiae = 0.0001
    # number of examples
    m = Y_hat.shape[0]
    derivative_cost = cost = None
    if (method is "cross_entropy"):
        if (derivative is None):
            # calculation of the cross entropy
            cost = (-1 / m) * (np.dot(Y.T, np.log(Y_hat + minutiae)) + 
                            np.dot(1 - Y.T, np.log(1 - Y_hat + minutiae)))
        else:
            # Calculation of first derivative
            derivative_cost = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    elif (method is "mean_sq_error"):
        if (derivative is None):
            # Calculation of mse
            cost = (-1 / m) * (np.dot((Y - Y_hat).T,(Y - Y_hat)))
        else:
            # Calculation of non-normalised first derivative
            derivative_cost = (-2 /m) * (Y - Y_hat)
    elif (method is "root_mean_sq_log_error"):
        if (derivative is None):
            # Calculation of rmsle
            cost = evaluate_rmsle(Y_hat, Y)
        else:
            grad_rmsle = grad(evaluate_rmsle)
            derivative_cost = -1 * grad_rmsle(Y_hat, Y)
    elif (method is "root_mean_sq_error"):
        if (derivative is None):
            # Calculation of rmse
            cost = evaluate_rmse(Y_hat, Y)
        else:
            grad_rmse = grad(evaluate_rmse)
            derivative_cost = -1 * grad_rmse(Y_hat, Y)
    else:
        raise Exception ("Error: Unknown cost method {}".format(method))
    return np.squeeze(cost), derivative_cost

# Stop training callback
# Cost is more important than accuracy, in training
g_contiguous_high_acc_epochs = 0
g_highest_acc_state = {"acc": 0.0, "epoch": 1}

def check_training_stop(state):
    global g_contiguous_high_acc_epochs
    global g_highest_acc_state
    accuracy = state["accuracy"]
    cost = state["cost"]
    epochs = state["epochs"]
    if (abs(cost) < COST_CUTOFF):
        g_contiguous_high_acc_epochs = g_contiguous_high_acc_epochs + 1
    else:
        g_contiguous_high_acc_epochs = 0
    # max acc check, disregard trivial values
    if (abs(accuracy) > 0.01 and abs(accuracy) < g_highest_acc_state["acc"]):
        g_highest_acc_state["acc"] = abs(accuracy)
        g_highest_acc_state["epoch"] = epochs
    stop = False
    if (g_contiguous_high_acc_epochs > MIN_CONTIGUOUS_HIGH_ACC_EPOCHS):
        stop = True
    # Prevent overfitting
    if(abs(accuracy) < 0.0195):
        stop = True
    return stop
# an auxiliary function that converts probability into class
# Poor man's sigmoid
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

"""
Kaggle house price Submissions are evaluated on Root-Mean-Squared-Error (RMSE) 
between the logarithm of the predicted value and the logarithm of the observed sales price
"""
def get_accuracy_value(Y_hat, Y, network_type="classifier"):
    if network_type == "classifier":
        Y_hat_ = convert_prob_into_class(Y_hat)
        acc = (Y_hat_ == Y).all(axis=0).mean()
    else:
        acc = evaluate_rmsle(Y_hat, Y)
        # If already kaggle Y is in log, so just use rmse
        # acc = evaluate_rmse(Y_hat, Y)
    return acc

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[0]
    
    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    # calculation of the activation function derivative
    dZ_curr = backward_activation_func(dA_curr, Z_curr)

    # derivative of the matrix W
    dW_curr = np.dot(dZ_curr.T, A_prev) / m
    # derivative of the vector b
    db_curr = np.sum(dZ_curr, keepdims=True) / m
    if (NN_DEBUG_SHAPES):
        print ("W_curr/dW_curr, b_curr/db_curr shape:", W_curr.shape, \
                dW_curr.shape, b_curr.shape, db_curr.shape)
    # derivative of the matrix A_prev
    dA_prev = np.dot(dZ_curr, W_curr.T)

    return dA_prev, dW_curr, db_curr

"""
input A --> L1 (A.W+b)--> Relu1 (A1) --> L2 (Z2=A1.W+b)--> Relu2 (A2)--> L3 (Z3=A2.W+b)--> Sigmoid --> Loss
Steps for backward propagation:
- Calculate derivative of loss function ??? But there is nothing to update in "loss" layer, or "sigmoid" layer
so is it derivative of L3*sigmoid*loss in one shot ???
- How to back propagate - 
        Find dError_Weight - error derivative wrt to that weight, 
        do Weight=(-dError_Weight*Weight)
- 
"""

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    
    # number of examples
    m = Y.shape[0]
    # initiation of gradient descent algorithm
    _, dA_prev = evaluate_cost_value(Y_hat, Y, NN_ARCHITECTURE_LOSS_TYPE, "first_derivative")
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        if (NN_DEBUG_SHAPES):        
            print ("shape dA/W/b/Z/A_prev = ",dA_curr.shape, W_curr.shape, \
                        b_curr.shape, Z_curr.shape, A_prev.shape)

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update_model(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        wRate = learning_rate * grads_values["dW" + str(layer_idx)]
        bRate = learning_rate * grads_values["db" + str(layer_idx)]
        if (NN_DEBUG_SHAPES):        
            print ("layer-name=",layer["layername"], " shape wRate/param_W/bRate/param_b = ", 
                wRate.shape, params_values["W" + str(layer_idx)].shape,
                bRate.shape, params_values["b" + str(layer_idx)].shape)
        params_values["W" + str(layer_idx)] -= wRate.T
        params_values["b" + str(layer_idx)] -= bRate

    return params_values

def train_model(X, Y, nn_architecture, epochs, learning_rate):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    cost_history = []
    accuracy_history = []
    train_state = {}
    
    # performing calculations for subsequent iterations
    global g_curr_epochs
    for i in range(epochs):
        g_curr_epochs = i
        # step forward
        # shuffle everytime
        merged = np.append(X,Y, axis=1)
        X_batch = np.array(X, copy=True)
        Y_batch = np.array(Y, copy=True)
        if NN_SHUFFLE_ROWS_EPOCH:
            np.random.shuffle(merged)
            X_batch = merged[0:,0:merged.shape[1]-1]
            Y_batch = merged[0:,-1].reshape(merged.shape[0],1)
        if NN_BATCH_ROWS_EPOCH:
            delete_n = int(X_batch.shape[0]/4)
            X_batch = X_batch[:-delete_n, :]
            Y_batch = Y_batch[:-delete_n, :]
        Y_hat, cache = evaluate_model(X_batch, params_values, nn_architecture)
        # calculating metrics and saving them in history
        cost, _ = evaluate_cost_value(Y_hat, Y_batch, NN_ARCHITECTURE_LOSS_TYPE)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y_batch, network_type=NN_TYPE)
        accuracy_history.append(accuracy)
        rms_error, _ = evaluate_cost_value(Y_hat, Y_batch, "root_mean_sq_error")
        
        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y_batch, cache, params_values, nn_architecture)
        # updating model state
        params_values = update_model(params_values, grads_values, nn_architecture, learning_rate)
        
        if(i % NN_DEBUG_PRINT_EPOCH_COUNT == 0):
            if(NN_DEBUG_VALUES):
                print("Epoch: {:06} - cost: {:.5f} - rms_error: {:.5f}".format(g_curr_epochs, cost, -rms_error))
        train_state["accuracy"] = accuracy
        train_state["cost"] = cost
        train_state["epochs"] = g_curr_epochs
        stop = check_training_stop(train_state)
        if (True == stop or NN_DEBUG_EXIT_EPOCH_ONE == True):
            print ("Breaking out of training, state = ", train_state)
            break
        global g_exit_signalled
        if g_exit_signalled > 0:
            break
    print_model(nn_architecture, params_values)
    print(g_highest_acc_state)
    return params_values

#############
if (NN_RUN_MODE == "separated_datapoints"):
    X, Y = generate_separated_datapoints(0, N_SAMPLES)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=NN_TEST_SIZE, random_state=42)
elif(NN_RUN_MODE == "kaggle_home"):
    X_mapping = {}  # Same mapping to be used in train/test !!
    X_train,X_train_normalize_state, X_mapping, Y_train, Y_train_normalize_state = \
        read_housing_csv("kaggle-housing-price/train2-outliers-heating-garagecars-poolqc-miscfeat-removed.csv", X_mapping, "SalePrice")
    X_test, X_test_normalize_state, X_mapping, Y_test, _ = \
        read_housing_csv("kaggle-housing-price/test2-heating-garagecars-poolqc-miscfeat-removed.csv", X_mapping)

# Training
input_size = X_train.shape[1]
arch = make_network_arch(input_size, network_type=NN_TYPE, network_shape=NN_SHAPE)
params_values = train_model(X_train, 
                    Y_train, arch, NN_EPOCHS, NN_LEARNING_RATE)

# Prediction
Y_test_hat, _ = evaluate_model(X_test, params_values, arch)

if NN_DEBUG_EXIT_EPOCH_ONE is True:
    print ("Exiting due to NN_DEBUG_EXIT_EPOCH_ONE...")
    exit(-1)

# Accuracy achieved on the test set
if (NN_RUN_MODE == "separated_datapoints"):
    acc_test = get_accuracy_value(Y_test_hat, Y_test, network_type=NN_TYPE)
    print("Numpy-twostick test accuracy: {:.2f}".format(acc_test))
else:
    Y_test_hat_denormalized = denormalize0(Y_test_hat, 
                    Y_train_normalize_state, NN_ZERO_MEAN_NORMALIZE)
    # Take anti-log of saleprice to get actuals, if log of price used for training
    # Y_test_hat_denormalized = np.exp(Y_test_hat_denormalized)
    Id = np.empty([2919-1461+1,1], dtype=int)
    for x in range(2919-1461+1):
        Id[x][0] = x + 1461
    Y_test_hat_denormalized = np.append(Id, Y_test_hat_denormalized,1)
    #print (Y_test_hat_denormalized)
    timestr = str(time.time())
    np.savetxt("submission-"+ timestr +".csv",Y_test_hat_denormalized, fmt="%d,%d", delimiter=",")
    np.save("params-"+ timestr +".csv",params_values)
    pklfile = open("params-"+ timestr +".pkl", 'wb')
    pickle.dump(params_values, pklfile)
    pklfile.close()
    print ("Model output and Parameters saved")
print ("Exiting ...")