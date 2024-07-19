# Imports
from torch import nn
import networkx as nx
from io import BytesIO
from matplotlib import pyplot as plt

def extract_sequential_network_data(mlp):
    """
    Extract the network structure and weights from a Pytorch-Sequential model.

    Args:
        mlp (torch.nn.Module): The model containing the Pytorch-Sequential network.

    Returns:
        tuple: A tuple containing:
            - network_structure (list of int): A list representing the structure of the network.
            - weights_list (list of numpy.ndarray): A list of weight matrices, where each matrix corresponds
              to a Linear layer in the network.
    """
    # Extract the Sequential model
    sequential_model = mlp.network

    # Initialize lists to store weights and layers
    weights_list = []
    layers = []

    # Iterate through the layers in the Sequential model
    for layer in sequential_model:
        if isinstance(layer, nn.Linear):
            # Extract weights and biases
            weights = layer.weight.detach().numpy().T  # Transpose to match the expected shape
            
            # Append the data to the lists
            weights_list.append(weights)
            layers.append(layer.out_features)

    # Define the network structure
    input_size = sequential_model[0].in_features  # Input size
    output_size = sequential_model[-2].out_features  # Output size, expecting the last layer not to be a Linear layer (e.g. Softmax)
    network_structure = [input_size] + layers[:-1] + [output_size]

    return network_structure, weights_list

def simple_mlp_vis(mlp, figsize=(20, 10), bottom2top=True):
    """
    Visualize a simple feedforward neural network using a directed graph.

    Args:
        mlp (torch.nn.Module): The model containing the Sequential network.
        figsize (tuple): Size of the figure (width, height) in inches.
        bottom2top (bool): If True, invert the y-axis to show the input layer at the bottom.

    Returns:
        numpy.ndarray: An RGB image of the network visualization.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Extract network data from the model
    network_structure, _ = extract_sequential_network_data(mlp)
    
    # Initialise dictionary to store neuron labels
    labels = {}
    
    # Determine the maximum number of neurons in any layer
    max_neurons = max(network_structure)
    
    # Add neurons to the graph
    for layer in range(len(network_structure)):
        # Get number of neurons in the current layer
        num_neurons = network_structure[layer]
        
        # Calculate a centering offset
        offset = (max_neurons - num_neurons) / 2
        
        for neuron in range(num_neurons):
            # Unique identifier for the neuron
            node_id = (layer, neuron)
            
            # Add the neuron to the graph
            G.add_node(node_id, layer=layer, pos=(neuron + offset, -layer))
            
            # Store the neuron label
            labels[node_id] = neuron
    
    # Add connections to the graph
    for layer in range(len(network_structure) - 1):
        for neuron1 in range(network_structure[layer]):
            for neuron2 in range(network_structure[layer + 1]):
                # Add a connection between the neurons
                G.add_edge((layer, neuron1), (layer + 1, neuron2))
    
    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph to the plot
    nx.draw(G, pos, labels=labels, with_labels=False, arrows=False, node_size=160, width=0.25,
            node_color=[(12/255, 142/255, 210/255)], edge_color=[(12/255, 142/255, 210/255)], edgecolors=[(0/255, 18/255, 30/255)])
    
    # Adjust plot limits to reduce space
    x_values, y_values = zip(*pos.values())
    x_margin = (max(x_values) - min(x_values)) * 0.05  # 5% margin
    y_margin = (max(y_values) - min(y_values)) * 0.05  # 5% margin
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Set the background color
    fig.set_facecolor((0/255, 18/255, 30/255))
    
    # Invert the y-axis if required
    if bottom2top:
        ax.invert_yaxis()
    
    # Save the plot to a BytesIO object (RAM)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    # Read the image from BytesIO and convert it to an RGB array
    img = plt.imread(buf)
    buf.close()
    plt.close(fig)
    
    # Transform the image to the range [0, 255]
    img = (img * 255).astype('uint8')
    
    return img