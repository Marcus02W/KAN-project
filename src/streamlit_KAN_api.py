import os
os.chdir('../KAN-Model')
from kan import KAN
from kan import MultKAN
import torch

dataset = torch.load('../data/MNIST/mnist_train_input.pth')
data_test = torch.load('../data/MNIST/mnist_test_input.pth')

def streamlit_kan_static_api(path: str) -> None:
    # params model
    hidden_1 = 3
    hidden_2 = 0
    steps = 45
    grid = 3
    k = 3
    seed = 42
    input_dim = dataset['train_input'].shape[ 1 ]   # Anzahl der Eingabefunktionen
    output_dim = 10   # Anzahl der Klasse
    #load model
    model = KAN(width=[input_dim, hidden_1, output_dim], grid= grid , k= k , seed= seed) 
    model = model.loadckpt(path)
    
