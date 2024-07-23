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

num_hidden_layers = range(1, 5)
hidden_sizes = [2**i for i in range(3, 7)]

for hidden_size in hidden_sizes:
    for num_hidden_layer in num_hidden_layers:
        # define network parameters
        input_size = len(x_train[0])
        num_hidden_layer = num_hidden_layer
        hidden_size = hidden_size
        output_size = train['label'].nunique()
        hidden_act = 'relu'
        output_act = 'softmax'
        dropout = 0.1

        # create model
        model = MLP(input_size, num_hidden_layer, hidden_size, output_size, hidden_act, output_act, dropout)
        
        # get the total number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # create a dataframe to safe information
        df = pd.DataFrame(columns=['num_hidden_layers', 'hidden_size', 'total_params', 'loss', 'time', 'accuracy'])
        df['num_hidden_layers'] = [num_hidden_layer]
        df['hidden_size'] = [hidden_size]
        df['total_params'] = [total_params]

        # train parameters
        batch_size = 1024
        loss_fn = 'cross_entropy'
        max_epochs = 64
        early_stop_threshold = 0.01
        early_stop_patience = 5
        lr = 0.001
        optimizer = 'adam'
        plot_loss = False
        return_loss = True

        # train model
        time_at_start = pd.Timestamp.now()
        loss = model.train(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer, plot_loss, return_loss)
        time_at_end = pd.Timestamp.now()
        df['loss'] = [loss]
        
        # save time information
        time = time_at_end - time_at_start
        hours, remainder = divmod(time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time = f'{hours}h:{minutes}m:{seconds}s'
        df['time'] = time

        # test model
        metric = 'accuracy'
        result = model.test(x_test, y_test, batch_size, metric)
        result = round(result[metric] * 100, 2)
        df['accuracy'] = result
        
        # create a directory to save the model and dataframe
        os.makedirs(f'./MLP/MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}', exist_ok=True)

        # save model
        model.save(f'./MLP/MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}/model.pth')
        
        # save dataframe
        df.to_csv(f'./MLP/MNIST_{num_hidden_layer}_{hidden_size}_{total_params}_{result}/info.csv', index=False)