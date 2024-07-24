# imports
import numpy as np
import pandas as pd
from torch import nn, Tensor, optim, no_grad, save as model_save, load as model_load, randperm, tensor, manual_seed
from torchsummary import summary as model_summary
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
import optuna

class MLP(nn.Module):
    """
    Multi-layer perceptron class.
    Designed to be used modularly with PyTorch.
    """
    def __init__(self, input_size: int, num_hidden_layers: int, hidden_size: int, output_size: int,
                 hidden_act: str, output_act: str, dropout: float) -> None:
        """
        Multi-layer perceptron constructor.
        
        Args:
            input_size (int): Number of input features.
            num_hidden_layers (int): Number of hidden layers.
            hidden_size (int): Number of neurons in each hidden layer.
            output_size (int): Number of output features.
            hidden_act (str): Activation function name for hidden layers (e.g., 'relu').
            output_act (str): Activation function name for output layer (e.g., 'softmax').
            dropout (float): Dropout rate.
        """
        # set seed
        manual_seed(42)
        
        # call parent constructor
        super(MLP, self).__init__()

        # variable declaration
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.dropout = dropout

        # input layer
        layers = [nn.Linear(input_size, hidden_size)]
        activation = self._get_activation(hidden_act)
        if activation:
            layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            activation = self._get_activation(hidden_act)
            if activation:
                layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # output layer
        layers.append(nn.Linear(hidden_size, output_size))
        activation = self._get_activation(output_act)
        if activation:
            layers.append(activation)

        self.network = nn.Sequential(*layers)

    def _get_activation(self, act_name: str) -> nn.Module:
        """
        Get activation function class from name.
        
        Args:
            act_name (str): Activation function name (e.g., 'relu').
        
        Returns:
            nn.Module: Activation function class.
        """
        if act_name == 'relu':
            return nn.ReLU()
        elif act_name == 'tanh':
            return nn.Tanh()
        elif act_name == 'sigmoid':
            return nn.Sigmoid()
        elif act_name == 'softmax':
            return nn.Softmax(dim=1)
        elif act_name == 'identity':
            return nn.Identity()
        else:
            raise ValueError(f"Activation function {act_name} not supported.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input data.
                
        Returns:
            Tensor: Output data.
        """
        return self.network(x)
    
    def forward_steps(self, input: np.array) -> list:
        """
        Step-by-step forward pass through the network.
        The activation states for input and after every activation function are observed.
        
        Args:
            x (Tensor): Input data.
                
        Returns:
            list: List of output data at each layer.
        """
        self.eval()
        # Check if input is 8x8 image
        if input.shape == (8, 8):
            # Flatten image
            input = input.flatten()
        
            # Tranform image array
            input =Tensor(input)
            
            # Flatten image
            input = input.view(-1)
            
            # Add batch dimension
            input = input.unsqueeze(0)
            
            # Initialize list to store steps and store input
            steps = []
            steps.append(input.squeeze().detach().numpy())
            
            # Iterate through layers and store output after each activation function
            for layer in self.network:
                input = layer(input)
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Identity):
                    steps.append(input.squeeze().detach().numpy())
            
            return steps
        else:
            # Error message if image shape is not 8x8
            raise ValueError("Image shape is not matching 8x8.")

    def summary(self) -> None:
        """Print network summary."""
        model_summary(self.network, (1, 1, self.input_size))

    def save(self, model_path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            model_path (str): Path to save model weights.
        """
        model_save(self, model_path)

    def fit(self, x: Tensor, y: Tensor, batch_size: int, loss_fn: str, max_epochs: int, early_stop_threshold: float,
              early_stop_patience: int, lr: float, optimizer: str, plot_loss: bool, return_loss: bool) -> None:
        """
        Training loop for the model.
        
        Args:
            x (Tensor): Input data.
            y (Tensor): Target data.
            batch_size (int): Number of samples per batch.
            loss_fn (str): Loss function name (e.g., 'mse').
            max_epochs (int): Maximum number of epochs to train.
            early_stop_threshold (float): Threshold for early stopping.
            early_stop_patience (int): Number of epochs to wait before early stopping.
            lr (float): Learning rate for optimizer.
            optimizer (str): Optimizer name (e.g., 'adam').
        """
        loss_fn_str = loss_fn
        self.train()
        
        def _plot_loss(loss_history: list, loss_fn: str) -> None:
            """
            Plot loss history.
            
            Args:
                loss_history (list): List of loss values.
                loss_fn (str): Loss function name (e.g., 'mse').
            """
            
            # plot loss history using matplotlib
            fig, ax = plt.subplots()
            ax.plot(loss_history)
            ax.set(xlabel='Epoch', ylabel=f'Loss ({loss_fn})', title='Training Loss')
            ax.grid()
            plt.show()
        
        # create dataset and dataloader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # choose loss function
        if loss_fn == 'mae':
            loss_fn = nn.L1Loss()
        elif loss_fn == 'mse':
            loss_fn = nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss function {loss_fn} not supported.")

        # choose optimizer
        if optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

        # set up early stopping parameters
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0

        # training loop through epochs
        progress_bar = tqdm(range(max_epochs), desc="Training epochs")
        for epoch in progress_bar:
            epoch_loss = 0.0

            # loop through batches
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                outputs = outputs.squeeze()
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # calculate epoch loss
            epoch_loss /= len(dataloader)
            progress_bar.set_postfix({'loss': epoch_loss})
            loss_history.append(epoch_loss)

            # early stopping check
            if epoch_loss < best_loss - early_stop_threshold:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Check if loss plot should be returned
        if return_loss: return loss_history
        
        # check if loss plot should be displayed
        if plot_loss: _plot_loss(loss_history, loss_fn_str)

    def test(self, x: Tensor, y: Tensor, batch_size: int, metric: str) -> dict:
        """
        Testing loop for the model.
        
        Args:
            x (Tensor): Input data.
            y (Tensor): Target data.
            batch_size (int): Number of samples per batch.
            metric (str): Metric name (e.g., 'accuracy').
                
        Returns:
            dict: Dictionary containing metric and value.
        """
        self.eval()
        
        # create dataset and dataloader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # define supported metrics and check if metric is supported
        supported_metrics = ['accuracy', 'precision', 'recall', 'f1', 'mae', 'mse', 'rmse', 'mape']
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported.")

        # get predictions
        with no_grad():
            y_pred = []
            for inputs, targets in dataloader:
                outputs = self.forward(inputs)
                outputs = outputs.squeeze()
                y_pred.extend(outputs.tolist())
            y_pred = Tensor(y_pred)
            
        # use logic for classification metrics
        if metric in ['accuracy', 'precision', 'recall', 'f1']:
            # use argmax to get class labels from probabilities
            y_pred = y_pred.argmax(dim=1)
            
            # calculate confusion matrix
            cm = confusion_matrix(y, y_pred)
        
        # choose metric calculation based on metric name
        match metric:
            case 'mae':
                value = mean_absolute_error(y, y_pred)
            case 'mse':
                value = mean_squared_error(y, y_pred)
            case 'rmse':
                value = root_mean_squared_error(y, y_pred)
            case 'mape':
                value = mean_absolute_percentage_error(y, y_pred)
            case 'accuracy':
                value = cm.diagonal().sum() / cm.sum()
            case 'precision':
                precision_values = {}
                for class_label in range(cm.shape[0]):
                    tp = cm[class_label, class_label]
                    fp = cm[:, class_label].sum() - tp
                    class_value = tp / (tp + fp)
                    precision_values[class_label] = class_value
                value = precision_values
            case 'recall':
                recall_values = {}
                for class_label in range(cm.shape[0]):
                    tp = cm[class_label, class_label]
                    fn = cm[class_label].sum() - tp
                    class_value = tp / (tp + fn)
                    recall_values[class_label] = class_value
                value = recall_values
            case 'f1':
                f1_values = {}
                for class_label in range(cm.shape[0]):
                    tp = cm[class_label, class_label]
                    fp = cm[:, class_label].sum() - tp
                    fn = cm[class_label].sum() - tp
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    class_value = 2 * (precision * recall) / (precision + recall)
                    f1_values[class_label] = class_value
                value = f1_values

        return {metric: value}
    
def mlp_tune_hyperparameters(x_train: Tensor, y_train: Tensor, input_size: int, num_hidden_layers: tuple, hidden_size: tuple,
                             output_size: int, hidden_act: str, output_act: str, dropout: tuple, batch_size: tuple,
                             loss_fn: str, max_epochs: tuple, early_stop_threshold: tuple, early_stop_patience: tuple,
                             lr: tuple, optimizer: str, plot_loss: bool, metric: str, opt_direction: str,
                             model_path: str, n_trials: int, split_ratio: float, study_name: str, return_loss: bool) -> MLP:
    """
    Hyperparameter tuning for the MLP model class.
    
    Args:
        input_size (int): Number of input features.
        num_hidden_layers (tuple): Min/Max tuple of number of hidden layers to try.
        hidden_size (tuple): Min/Max tuple of number of neurons in each hidden layer to try.
        output_size (int): Number of output features.
        hidden_act (str): Activation function name for hidden layers (e.g., 'relu').
        output_act (str): Activation function name for output layer (e.g., 'softmax').
        dropout (tuple): Min/Max tuple of dropout rates to try.
        batch_size (tuple): Min/Max tuple of batch sizes to try.
        loss_fn (str): Loss function name (e.g., 'mse').
        max_epochs (tuple): Min/Max tuple of maximum number of epochs to train.
        early_stop_threshold (tuple): Min/Max tuple of thresholds for early stopping to try.
        early_stop_patience (tuple): Min/Max tuple of number of epochs to wait before early stopping to try.
        lr (tuple): Min/Max tuple of learning rates to try.
        optimizer (str): Optimizer name (e.g., 'adam').
        plot_loss (bool): Plot loss history.
        metric (str): Metric name (e.g., 'accuracy').
        opt_direction (str): Optimization direction ('minimize' or 'maximize').
        model_path (str): Path to save best model.
        n_trials (int): Number of trials for hyperparameter tuning.
        split_ratio (float): Ratio of training data to use for validation.
        study_name (str): Name of the optuna study.
        return_loss (bool): Return loss history.
        
    Returns:
        MLP: Best model found during hyperparameter tuning.
    """
    # Random sample validation set
    random_indices = randperm(x_train.size(0))[:round(len(x_train) * split_ratio)]
    x_val = x_train[random_indices]
    y_val = y_train[random_indices]
    
    remaining_indices = tensor([i for i in range(len(x_train)) if i not in random_indices])
    x_train = x_train[remaining_indices]
    y_train = y_train[remaining_indices]
    
    def _objective(trial: optuna.Trial, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                     input_size=input_size, num_hidden_layers=num_hidden_layers, hidden_size=hidden_size,
                     output_size=output_size, hidden_act=hidden_act, output_act=output_act, dropout=dropout,
                     batch_size=batch_size, loss_fn=loss_fn, max_epochs=max_epochs, early_stop_threshold=early_stop_threshold,
                     early_stop_patience=early_stop_patience, lr=lr, optimizer=optimizer, plot_loss=plot_loss, metric=metric,
                     opt_direction=opt_direction, model_path=model_path) -> float:
        """
        Objective function for hyperparameter tuning.
        
        Args:
            trial (optuna.Trial): Optuna trial object.
            (...) Downstream arguments are passed for optimization.
            
        Returns:
            float: Metric value to minimize.
        """
        # best metric initialization
        if opt_direction == 'minimize':
            best_metric = float('inf')
        elif opt_direction == 'maximize':
            best_metric = 0.0
        
        # sample hyperparameters
        num_hidden_layers = trial.suggest_int('num_hidden_layers', num_hidden_layers[0], num_hidden_layers[1])
        hidden_size = trial.suggest_int('hidden_size', hidden_size[0], hidden_size[1])
        dropout = trial.suggest_float('dropout', dropout[0], dropout[1])
        batch_size = trial.suggest_int('batch_size', batch_size[0], batch_size[1])
        max_epochs = trial.suggest_int('max_epochs', max_epochs[0], max_epochs[1])
        early_stop_threshold = trial.suggest_float('early_stop_threshold', early_stop_threshold[0], early_stop_threshold[1])
        early_stop_patience = trial.suggest_int('early_stop_patience', early_stop_patience[0], early_stop_patience[1])
        lr = trial.suggest_float('lr', lr[0], lr[1])
        
        # create model
        model = MLP(input_size, num_hidden_layers, hidden_size, output_size, hidden_act, output_act, dropout)
        
        # train model
        model.fit(x_train, y_train, batch_size, loss_fn, max_epochs, early_stop_threshold, early_stop_patience, lr, optimizer, plot_loss, return_loss)
        
        # test model
        result = model.test(x_val, y_val, batch_size, metric)
        
        # update best metric and model
        if opt_direction == 'minimize':
            if result[metric] < best_metric:
                best_metric = result[metric]
                model_save(model, model_path)
        elif opt_direction == 'maximize':
            if result[metric] > best_metric:
                best_metric = result[metric]
                model_save(model, model_path)
        
        return result[metric]
    
    # create optuna study and optimize
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(direction=opt_direction, study_name=study_name ,storage=storage_name, load_if_exists=True)
    study.optimize(_objective, n_trials, show_progress_bar=True)
    
    return model_load(model_path)

def mnist_forward_df(model: MLP, input: np.array) -> pd.DataFrame:
    """
    Helper function to visualize the forward pass of the MNIST MLP model in our app.
    
    Args:
        model (MLP): Trained MLP model.
        input (np.array): Input image array.
        
    Returns:
        DataFrame: DataFrame containing output probabilities. Data is stored in 'probabilities' column.
    """
    
    def softmax(x):
        return x / x.sum(axis=0)
    
    # check if the np image array is 8x8
    if input.shape == (8, 8):
        
        # Forward pass
        forward_steps = model.forward_steps(input)
        output = forward_steps[-1]
        
        # Normalize output
        min_val = output.min()
        max_val = output.max()
        output = (output - min_val) / (max_val - min_val)
        
        # Apply Softmax to logits
        output = softmax(output)
        
        # Create DataFrame
        output = pd.DataFrame(output, columns=['probabilities'])
        
        return output
    else:
        # Error message if image shape is not 8x8
        raise ValueError("Image shape is not matching 8x8.")