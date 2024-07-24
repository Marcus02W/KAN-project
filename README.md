# KAN-project

This project aims at developing a webapp that allows for a comparison between the newly published Kolmogorov-Arnold-Networks (KANs) and traditional MLPs trained on MNIST and custom datasets by providing visual insights into the architectural behaviour of the networks.

## Data Downloads

Download and paste datasets in the "data" directory in their respective subdirectory.

### MNIST

[Kaggle Download](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data "Link to visit the MNIST dataset download page on Kaggle")

### California Housing

[Kaggle Download](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data "Link to visit the California Housing dataset download page on Kaggle")

## Install Requirements

Use/Install Python 3.11.9 with (pre)installed Virtual Environments package or an (ana)conda installation to create the environment.

### Virtual Envorinment

1. Create

```bash
python -m venv kan
```

2. Activate

Unix/Linux/Max:

```bash
source kan/bin/activate
```

Windows:

```
kan\Scripts\activate
```

3. Install

```python
pip install -r requirements.txt
```

### Conda Environment

1. Create

```bash
conda create --name kan python=3.11.9
```

2. Activate

```bash
conda activate kan
```

3. Install

```bash
pip install -r requirements.txt
```

### Important Data in Repository
All important files for the webapp functionality are in the src folder as backend and frontend can be found in the Streamlit folder.
The folder contains some notebooks with tests and the HPO process. In addition, the version of the pykan github from 17.07.2024 can be found in the kan folder. In the models folder are the finished KAN and MLP models.
 
### KAN HPO WANDB
Here the Link to the WANDB HPO process: https://wandb.ai/lukasgrengames/


### How to run streamlit
1. cd into the "streamlit" folder
2. run command: "streamlit run website.py"

###
Fabian Banovic, Joschua Jakubek, Lukas Bruckner, Marcus Wirth
