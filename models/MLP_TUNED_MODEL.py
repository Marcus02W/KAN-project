# local imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from MLP import MLP, mlp_tune_hyperparameters

# global imports
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
import optuna

# set seed
torch.manual_seed(42)

# read train and test data
train = pd.read_csv('../data/MNIST/mnist_train.csv')
test = pd.read_csv('../data/MNIST/mnist_test.csv')

# convert data to numpy arrays
x_train = train.drop('label', axis=1).values
x_test = test.drop('label', axis=1).values
y_train = train['label'].values
y_test = test['label'].values

# set target size to scale images
target_size = 8

# define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor()
])

# transform the data
def transform_images(images):
    transformed_images = []
    for image in images:
        image = image.reshape(28, 28).astype(np.uint8)
        image = transform(image)
        image = image.view(-1)
        transformed_images.append(image)
    return torch.stack(transformed_images)

x_train = transform_images(x_train)
x_test = transform_images(x_test)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

# load hyperparameters from optuna database file
study = optuna.load_study(study_name='MLP-MNIST', storage='sqlite:///../demos/MLP-MNIST.db')
best_params = study.best_trial.params

# define network parameters
input_size = len(x_train[0])
num_hidden_layer = best_params['num_hidden_layers']
hidden_size = best_params['hidden_size']
output_size = train['label'].nunique()
hidden_act = 'relu'
output_act = 'identity'
dropout = best_params['dropout']

# create model
model = MLP(input_size, num_hidden_layer, hidden_size, output_size, hidden_act, output_act, dropout)

# get the total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# create a dataframe to safe information
df = pd.DataFrame()
df['num_hidden_layers'] = [num_hidden_layer]
df['hidden_size'] = [hidden_size]
df['total_params'] = [total_params]

# train parameters
batch_size = best_params['batch_size']
loss_fn = 'cross_entropy'
max_epochs = best_params['max_epochs']
early_stop_threshold = best_params['early_stop_threshold']
early_stop_patience = best_params['early_stop_patience']
lr = best_params['lr']
optimizer = 'adam'
plot_loss = False
return_loss = True

# train model
time_at_start = pd.Timestamp.now()
loss = model.fit(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer, plot_loss, return_loss)
time_at_end = pd.Timestamp.now()
df['train_loss_history'] = [loss]
df['train_loss'] = df['train_loss_history'][0][-1]

# save time information
time = time_at_end - time_at_start
hours, remainder = divmod(time.seconds, 3600)
minutes, seconds = divmod(remainder, 60)
time = f'{hours}h:{minutes}m:{seconds}s'
df['time'] = time

# get the loss value on the test set
ce_loss = CrossEntropyLoss()
preds = model(x_test)
loss = ce_loss(preds, y_test)
df['test_loss'] = loss.item()

# test model
metric = 'accuracy'
result = model.test(x_test, y_test, batch_size, metric)
result = round(result[metric] * 100, 2)
df['accuracy'] = result

# create a directory to save the model and dataframe
os.makedirs(f'./MLP/TUNED_MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}', exist_ok=True)

# save model
model.save(f'./MLP/TUNED_MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}/model.pth')

# save dataframe
df.to_csv(f'./MLP/TUNED_MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}/info.csv', index=False)