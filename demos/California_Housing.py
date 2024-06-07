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
batch_size = 100
loss_fn = 'mse'
max_epochs = 256
early_stop_threshold = 1000
early_stop_patience = 8
lr = 0.001
optimizer = 'adam'

# train model
model.train(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer)

# test model
metric = 'mae'
result = model.test(x_test, y_test, batch_size, metric)

# print result
print(result)