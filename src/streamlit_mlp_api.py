from torch import load
from MLP import mnist_forward_df
from MLP_VIS import mlp_weights_vis, mlp_forward_vis

def streamlit_mlp_static_api(path: str) -> None:
    
    model = load(path)
    
    img = mlp_weights_vis(model)
    
    return img

def streamlit_mlp_inference_api(path: str, image) -> None:
    
    model = load(path)
    
    pred_df = mnist_forward_df(model, image)
    
    img = mlp_inference_vis(model, image)
    
    return pred_df, img