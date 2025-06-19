import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def gen_data(p=19):
    """
    Returns:
      inputs: shape (p**2, 2*p), all possible pairs (num1, num2).
      labels: shape (p**2,), where labels[i] = (num1 + num2) mod p.
    """
    data_inputs = []
    data_labels = []
    for num1 in range(p):
        for num2 in range(p):
            # One-hot encode each digit
            enc1 = torch.zeros(p)
            enc1[num1] = 1.0
            enc2 = torch.zeros(p)
            enc2[num2] = 1.0

            # Concatenate
            inp_vec = torch.cat([enc1, enc2], dim=0)
            # Label is (num1 + num2) mod p
            label = (num1 + num2) % p

            data_inputs.append(inp_vec)
            data_labels.append(label)

    inputs = torch.stack(data_inputs)  # shape (p**2, 2*p)
    labels = torch.tensor(data_labels) # shape (p**2,)
    return inputs, labels

def split_train_validation(inputs, labels, proportion=0.6, seed=42):
    """
    Splits the data into train and validation sets.
    
    Args:
      inputs: Tensor of shape (N, D)
      labels: Tensor of shape (N,)
      proportion: float in [0, 1], fraction of data to use for training
      seed: for reproducibility
      
    Returns:
      (train_inputs, train_labels, val_inputs, val_labels)
    """
    N = inputs.shape[0]
    indices = torch.arange(N)
    rng = torch.Generator().manual_seed(seed)
    shuffled = indices[torch.randperm(N, generator=rng)]

    train_size = int(proportion * N)
    train_indices = shuffled[:train_size]
    val_indices   = shuffled[train_size:]

    train_inputs = inputs[train_indices]
    train_labels = labels[train_indices]
    val_inputs   = inputs[val_indices]
    val_labels   = labels[val_indices]

    return train_inputs, train_labels, val_inputs, val_labels

def sorted_inputs(p):
    """"
    Returns sorted inputs based on the modulo sum of the two input numbers.
    """
    inputs, labels = gen_data(p)
    sorted_indices = torch.argsort(labels)
    sorted_inputs = inputs[sorted_indices]
    sorted_labels = labels[sorted_indices]
    return sorted_inputs, sorted_labels

def sorted_inputs_display(p):
    """
    Returns sorted inputs based on the modulo sum of the two input numbers.
    """
    inputs, labels = sorted_inputs(p)
    input_display = []
    for i in range(len(inputs)):
        num1 = torch.argmax(inputs[i][:p]).item()
        num2 = torch.argmax(inputs[i][p:]).item()
        input_display.append(f"{num1}+{num2}={(num1+num2)%p}")
    return input_display

# ------------------------------------------------------------------
# 2) Define the network architecture
# ------------------------------------------------------------------
class ModuloAdditionNet(nn.Module):
    def __init__(self, p, number_of_hidden_units=10): #int((p + 1) * p / 2)
        super().__init__()
        self.fc1 = nn.Linear(2 * p, number_of_hidden_units, bias=True) #simple network with 1 hidden layer with number_of_hidden_units
        self.fc2 = nn.Linear(number_of_hidden_units, p, bias=False)


    def forward(self, x):
        z1 = self.fc1(x)
        a1 = F.relu(z1)
        z2 = self.fc2(a1)
        return z2
    

#load in the saved weights data
def load_weights(model, path):
    """
    Load the model weights from a file.
    
    Args:
      model: The model to load weights into.

    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Weights loaded from {path}")
    else:
        print(f"No weights found at {path}. Using default initialization.")
    return model

def extract_weights(model):
    """
    Extract weights and biases from the model.
    
    Args:
      model: The model to extract weights from.
      
    Returns:
      W_1: Weights of the first layer.
      W_2: Weights of the second layer.
      b: Biases of the first layer.
    """
    W_1 = model.fc1.weight.data
    W_2 = model.fc2.weight.data
    b = model.fc1.bias.data
    return W_1.numpy(), b.numpy(), W_2.numpy()
