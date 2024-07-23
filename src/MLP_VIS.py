# Imports
from torch import nn
import networkx as nx
from io import BytesIO
from matplotlib import pyplot as plt
from interpolate_color import interpolate_color

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

def mlp_weights_vis(mlp, figsize=(20, 10), bottom2top=True):
    """
    Visualize a simple feedforward neural network using a directed graph with edge transparency
    based on the normalized weights.

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
    network_structure, weights_list = extract_sequential_network_data(mlp)
    
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
    
    # Normalize weights and add connections to the graph with edge transparency based on weights
    for layer in range(len(weights_list)):
        weights = weights_list[layer]
        # Normalize weights between 0 and 1
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        for neuron1 in range(weights.shape[0]):
            for neuron2 in range(weights.shape[1]):
                # Get the normalized weight for the edge transparency
                transparency = norm_weights[neuron1, neuron2]
                G.add_edge((layer, neuron1), (layer + 1, neuron2), weight=transparency)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph to the plot with edge transparency based on weights
    edges = G.edges(data=True)
    edge_colors = [(12/255, 142/255, 210/255, edge[2]['weight']) for edge in edges]
    nx.draw(G, pos, labels=labels, with_labels=False, arrows=False, node_size=160, node_color=[(12/255, 142/255, 210/255)], edge_color=edge_colors, edgecolors=[(0/255, 18/255, 30/255)], width=0.6)
    
    # Adjust plot limits to reduce space
    x_values, y_values = zip(*pos.values())
    x_margin = (max(x_values) - min(x_values)) * 0.05  # 5% margin
    y_margin = (max(y_values) - min(y_values)) * 0.05  # 5% margin
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Set the background color
    fig.set_facecolor((14/255, 17/255, 23/255))
    
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

def mlp_inference_vis(mlp, input_image, figsize=(20, 10), bottom2top=True):
    """
    Visualize a simple feedforward neural network using a directed graph with edge transparency
    based on the normalized weights and node colors based on the activations.

    Args:
        mlp (torch.nn.Module): The model containing the Sequential network.
        input_image (numpy.ndarray): Input image to pass through the network.
        figsize (tuple): Size of the figure (width, height) in inches.
        bottom2top (bool): If True, invert the y-axis to show the input layer at the bottom.

    Returns:
        numpy.ndarray: An RGB image of the network visualization.
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Extract network data from the model
    network_structure, weights_list = extract_sequential_network_data(mlp)
    
    # Get the activation values from the forward steps
    activations = mlp.forward_steps(input_image)
    
    # Normalize activation values for each layer between 0 and 1
    normalized_activations = []
    for act in activations:
        min_val = act.min()
        max_val = act.max()
        if max_val - min_val > 0:
            normalized_activations.append((act - min_val) / (max_val - min_val))
        else:
            normalized_activations.append(act)
    
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
            
            # Determine the node color based on the activation value
            
            activation_value = normalized_activations[layer][neuron]
            node_color = interpolate_color((12, 142, 210), (186, 0, 32), activation_value)
            node_color = tuple(val / 255 for val in node_color)
            
            # Add the neuron to the graph with its color
            G.add_node(node_id, layer=layer, pos=(neuron + offset, -layer), color=node_color)
            
            # Store the neuron label
            labels[node_id] = neuron
    
    # Normalize weights and add connections to the graph with edge transparency based on weights
    for layer in range(len(weights_list)):
        weights = weights_list[layer]
        # Normalize weights between 0 and 1
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        for neuron1 in range(weights.shape[0]):
            for neuron2 in range(weights.shape[1]):
                # Get the normalized weight for the edge transparency
                transparency = norm_weights[neuron1, neuron2]
                G.add_edge((layer, neuron1), (layer + 1, neuron2), weight=transparency)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions and colors
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = [G.nodes[node]['color'] for node in G.nodes]
    
    # Draw the graph to the plot with edge transparency based on weights
    edges = G.edges(data=True)
    edge_colors = [(12/255, 142/255, 210/255, edge[2]['weight']) for edge in edges]
    nx.draw(G, pos, labels=labels, with_labels=False, arrows=False, node_size=160, node_color=node_colors, edge_color=edge_colors, edgecolors=[(0/255, 18/255, 30/255)], width=0.6)
    
    # Adjust plot limits to reduce space
    x_values, y_values = zip(*pos.values())
    x_margin = (max(x_values) - min(x_values)) * 0.05  # 5% margin
    y_margin = (max(y_values) - min(y_values)) * 0.05  # 5% margin
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Set the background color
    fig.set_facecolor((14/255, 17/255, 23/255))
    
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