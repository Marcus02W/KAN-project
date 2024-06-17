# local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# global imports
from MLP import MLP, mlp_tune_hyperparameters
import pandas as pd
import torch

# set seed
torch.manual_seed(42)

# read train and test data
train = pd.read_csv('../data/MNIST/mnist_train.csv')
test = pd.read_csv('../data/MNIST/mnist_test.csv')

# convert data to tensors
x_train = torch.tensor(train.drop('label', axis=1).values, dtype=torch.float32)
x_test = torch.tensor(test.drop('label', axis=1).values, dtype=torch.float32)
y_train = torch.tensor(train['label'].values, dtype=torch.int64)
y_test = torch.tensor(test['label'].values, dtype=torch.int64)

# set mode ('train', 'opt', 'load)
mode = 'train'

# regular training
if mode == 'train':
    # define network parameters
    input_size = len(x_train[0])
    num_hidden_layers = 8
    hidden_size = 16
    output_size = train['label'].nunique()
    hidden_act = 'relu'
    output_act = 'softmax'
    dropout = 0.1

    # create model
    model = MLP(input_size, num_hidden_layers, hidden_size, output_size, hidden_act, output_act, dropout)
    model.summary()

    # train parameters
    batch_size = 1024
    loss_fn = 'cross_entropy'
    max_epochs = 64
    early_stop_threshold = 0.01
    early_stop_patience = 4
    lr = 0.001
    optimizer = 'adam'
    plot_loss = True

    # train model
    model.train(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer, plot_loss)

    # test model
    metric = 'accuracy'
    result = model.test(x_test, y_test, batch_size, metric)

    # print result
    print(result)

    # save model
    model.save('../models/MNIST_MLP.pth')

# hyperparameter optimization    
elif mode == 'opt':
    # define network parameters
    input_size = len(x_train[0])
    output_size = train['label'].nunique()
    hidden_act = 'relu'
    output_act = 'softmax'
    loss_fn = 'cross_entropy'
    optimizer = 'adam'
    plot_loss = False
    metric = 'accuracy'
    model_path = '../models/MNIST_MLP_OPT.pth'
    
    # define optimization ranges
    num_hidden_layers = (2, 16)
    hidden_size = (4, 32)
    dropout = (0.05, 0.25)
    batch_size = (100, 10000)
    max_epochs = (16, 128)
    early_stop_threshold =(0.001, 0.1)
    early_stop_patience = (2, 8)
    lr = (0.0001, 0.1)
    
    # define optuna parameters
    opt_direction = 'maximize'
    num_trials = 10
    
    # call optimization function
    tuned_model = mlp_tune_hyperparameters(x_train, y_train, x_test, y_test, input_size,
                             num_hidden_layers, hidden_size, output_size, hidden_act, output_act,
                             dropout, batch_size, loss_fn, max_epochs, early_stop_threshold,
                             early_stop_patience, lr, optimizer, plot_loss, metric,
                             opt_direction, model_path, num_trials)
    
# loading and testing model    
elif mode == 'load':
    # load model
    model = torch.load('../models/MNIST_MLP_OPT.pth')
    
    # test model
    batch_size = 1024
    metric = 'accuracy'
    result = model.test(x_test, y_test, batch_size, metric)
    
    # print result
    print(result)