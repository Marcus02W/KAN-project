# imports
from torch import nn, Tensor, optim, no_grad, max as torch_max, save as model_save, load as model_load
from torchsummary import summary as model_summary
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

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

    def summary(self) -> None:
        """Print network summary."""
        model_summary(self.network, (1, 1, self.input_size))

    def save(self, model_path: str) -> None:
        """
        Save model weights to disk.
        
        Args:
            model_path (str): Path to save model weights.
        """
        model_save(self.network, model_path)

    def load(self, model_path: str) -> None:
        """
        Load model weights from disk.
        
        Args:
            model_path (str): Path to load model weights.
        """
        self.network = model_load(model_path)

    def train(self, x: Tensor, y: Tensor, batch_size: int, loss_fn: str, max_epochs: int,
              early_stop_threshold: float, early_stop_patience: int, lr: float, optimizer: str, plot_loss: bool) -> None:
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