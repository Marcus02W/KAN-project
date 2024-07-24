import torch
from KAN_VIS_statik import kan_weights_vis
from KAN_for import df_foward_kan

import os
os.chdir('../KAN-Model')
from kan_local import KAN
from kan_local import MultKAN

dataset = torch.load('../data/MNIST/mnist_train_input.pth')
data_test = torch.load('../data/MNIST/mnist_test_input.pth')

def streamlit_kan_static_api(path: str,hidden_1: int, hidden_2: int) -> None:
    # params model
    hidden_1 = hidden_1
    hidden_2 = hidden_2
    input_dim = dataset['train_input'].shape[ 1 ]   # Anzahl der Eingabefunktionen
    output_dim = 10   # Anzahl der Klasse
    #load model
    if hidden_2 != 0:   
        model = KAN(width=[input_dim, hidden_1, hidden_2, output_dim], grid= 3 , k= 3 , seed= 42)
    else:
        model = KAN(width=[input_dim, hidden_1, output_dim], grid= 3 , k= 3 , seed= 42) 
    model = model.loadckpt(path)
    img = kan_weights_vis(model,1.5)
    
    return img
    
def streamlit_kan_inference_api(path: str, image, hidden_1: int, hidden_2: int) -> None:
    
    # params model
    hidden_1 = hidden_1
    hidden_2 = hidden_2
    input_dim = dataset['train_input'].shape[ 1 ]   # Anzahl der Eingabefunktionen
    output_dim = 10   # Anzahl der Klasse
    #load model
    if hidden_2 != 0:   
        model = KAN(width=[input_dim, hidden_1, hidden_2, output_dim], grid= 3 , k= 3 , seed= 42)
    else:
        model = KAN(width=[input_dim, hidden_1, output_dim], grid= 3 , k= 3 , seed= 42) 
    model = model.loadckpt(path)
    
    pred_df, img = df_foward_kan(model, image,dataset,1.5)
    
    return pred_df, img