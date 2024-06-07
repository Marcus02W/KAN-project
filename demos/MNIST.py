# local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

# global imports
from MLP import MLP
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

# define network parameters
input_size = len(x_train[0])
num_hidden_layers = 8
hidden_size = 32
output_size = train['label'].nunique()
hidden_act = 'relu'
output_act = 'softmax'
dropout = 0.1

# create model
model = MLP(input_size, num_hidden_layers, hidden_size, output_size, hidden_act, output_act, dropout)
model.summary()

# train parameters
batch_size = 10000
loss_fn = 'cross_entropy'
max_epochs = 64
early_stop_threshold = 0.005
early_stop_patience = 3
lr = 0.001
optimizer = 'adam'

# train model
model.train(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer)

# test model
metric = 'accuracy'
result = model.test(x_test, y_test, batch_size, metric)

# print result
print(result)

# save model
model.save('../models/MNIST_MLP.pth')