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

# read data and fill missing values
data = pd.read_csv('../data/California_Housing/housing.csv')
data = data.fillna(0)

# one hot encoding for ocean_proximity
data = pd.get_dummies(data, columns=['ocean_proximity'], dtype=int)

# train test split
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

# convert data to tensors
x_train = torch.tensor(train.drop('median_house_value', axis=1).values, dtype=torch.float32)
x_test = torch.tensor(test.drop('median_house_value', axis=1).values, dtype=torch.float32)
y_train = torch.tensor(train['median_house_value'].values, dtype=torch.float32)
y_test = torch.tensor(test['median_house_value'].values, dtype=torch.float32)

# set mode ('train', 'opt', 'load)
mode = 'train'

# regular training
if mode == 'train':
    # define network parameters
    input_size = len(x_train[0])
    num_hidden_layers = 8
    hidden_size = 16
    output_size = 1
    hidden_act = 'relu'
    output_act = 'relu'
    dropout = 0.1

    # create model
    model = MLP(input_size, num_hidden_layers, hidden_size, output_size, hidden_act, output_act, dropout)
    model.summary()

    # train parameters
    batch_size = 128
    loss_fn = 'mae'
    max_epochs = 128
    early_stop_threshold = 1000
    early_stop_patience = 8
    lr = 0.001
    optimizer = 'adam'
    loss_plot = True

    # train model
    model.train(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer, loss_plot)

    # test model
    metric = 'mape'
    result = model.test(x_test, y_test, batch_size, metric)

    # print result
    print(result)
    
    # save model
    model.save('../models/California_Housing_MLP.pth')

# hyperparameter optimization
elif mode == 'opt':
    # define network parameters
    input_size = len(x_train[0])
    output_size = 1
    hidden_act = 'relu'
    output_act = 'relu'
    loss_fn = 'mae'
    optimizer = 'adam'
    plot_loss = False
    metric = 'mape'
    model_path = '../models/California_Housing_MLP_OPT.pth'
    
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
    opt_direction = 'minimize'
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
    model = torch.load('../models/California_Housing_MLP.pth')
    
    # test model
    batch_size = 1024
    metric = 'mape'
    result = model.test(x_test, y_test, batch_size, metric)
    
    # print result
    print(result)