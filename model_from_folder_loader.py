import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import copy


from utility_functions import gen_data, split_train_validation, sorted_inputs, ModuloAdditionNet, load_weights, extract_weights
from invariance_and_analysis_functions import apply_invariance_remove_b, apply_invariance_scalling, generate_heatmaps


# progress measures
def activation_sparsity(train_inputs, p, number_of_hidden_units, tau, model):
    """
    Calculate the activation sparsity of the model on the training inputs.
    
    Args:
      train_inputs: Tensor of shape (N, D)
      model: The model to evaluate.

    Returns:
      sparsity: float, the activation sparsity.
    """
    with torch.no_grad():
        z1 = model.fc1(train_inputs).numpy()
        a1 = F.relu(torch.tensor(z1)).numpy()
        sparsity = np.mean(np.sum(a1 < tau, axis=1) / number_of_hidden_units)
    return sparsity

def absolute_weight_entropy(model):
    """
    Calculate the absolute weight entropy of the model.
    
    Args:
      model: The model to evaluate.

    Returns:
      entropy: float, the absolute weight entropy.
    """
    with torch.no_grad():
        weights = model.fc1.weight.numpy()
        abs_weights = np.abs(weights)
        entropy = -np.sum(abs_weights * np.log(abs_weights + 1e-10))  # Add small value to avoid log(0)
    return entropy

def approximate_local_circuit_complexity(model, inputs, seed):
    """
    Calculate the approximate local circuit complexity of the model.
    
    Args:
      model: The model to evaluate.

    Returns:
      complexity: float, the approximate local circuit complexity.
    """
    np.random.seed(seed)
    with torch.no_grad():
        #first we calculate the original logits
        original_logits = model(inputs)
        #we set 10% of the weights to 0 in W_1
        W_1 = model.fc1.weight.numpy()
        W_2 = model.fc2.weight.numpy()
        b = model.fc1.bias.numpy()
        number_of_weights_W_1 = W_1.size
        W_1_random_seed = np.random.uniform(0, 1, number_of_weights_W_1)
        W_1_random_seed = W_1_random_seed.reshape(W_1.shape)
        W_1[W_1_random_seed < 0.1] = 0
        #new logits
        W_1 = torch.tensor(W_1)
        W_2 = torch.tensor(W_2)
        b = torch.tensor(b)
        logits = torch.matmul(inputs, W_1.T) + b
        logits = F.relu(logits)
        logits = torch.matmul(logits, W_2.T)

        # softmax
        soft_max_logits = F.softmax(logits, dim=1)
        original_soft_max_logits = F.softmax(original_logits, dim=1)
        # calculate the KL divergence
        kl_div = torch.nn.KLDivLoss(reduction='sum')
        kl_div_loss = kl_div(torch.log(soft_max_logits), original_soft_max_logits)
        
    return kl_div_loss


def L_infty_norm(model):
    """
    Calculate the L_infinity norm of the model.
    
    Args:
      model: The model to evaluate.

    Returns:
      norm: float, the L_infinity norm.
    """
    with torch.no_grad():
        W_1 = model.fc1.weight.numpy()
        W_2 = model.fc2.weight.numpy()
        b = model.fc1.bias.numpy()
        for i in range(len(b)):
          a_i = 0.5 * b[i]
          W_1[i] = W_1[i] + a_i * np.ones(W_1.shape[1])
          norm_i = np.max(np.abs(W_1[i]))
          W_2[:,i] = W_2[:,i] * norm_i

        # From the invariance properties we know that we can normalize the weights with respect to the infinity norm, for each row of W_1 and column of W_2, so we only have keep track of one of the two matrices.

        
    return np.max(np.abs(W_2))

def similarity(model, inputs_sorted, p):
    """
    Calculate the similarity of the Relu(W1@X+b) according to Definition 5.20.
    
    Args:
      model: The model to evaluate.
      inputs_sorted: The inputs to evaluate (sorted by modular sum).
      p: The modulo base.

    Returns:
      similarity: float, the similarity measure.
    """
    model.eval()
    W_1_orig, b_1_orig, W_2_orig = extract_weights(model)
    
    # Apply the same invariances as before
    W_1_no_bias, _ = apply_invariance_remove_b(W_1_orig, W_2_orig, b_1_orig)
    W_1_norm, W_2_norm = apply_invariance_scalling(
        W_1_no_bias, W_2_orig.copy(), b=None, type='l1', matrix='W_1'
    )
    
    print("Calculating similarity measure...")
    
    # Convert inputs to numpy if they're tensors
    if isinstance(inputs_sorted, torch.Tensor):
        inputs_np = inputs_sorted.detach().numpy()
    else:
        inputs_np = inputs_sorted
    
    # Directly compute activations with numpy: ReLU(X @ W_1^T)
    layer_1_outputs = np.maximum(0, np.dot(inputs_np, W_1_norm.T))
    
    # Divide layer_1_outputs into p parts
    layer_1_outputs_parts = np.split(layer_1_outputs, p, axis=0)
    
    # Calculate similarity as before
    sum_dif_min = 0
    for part in layer_1_outputs_parts:
        abs_difs = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1, p):
                abs_difs[i, j] = np.sum(np.abs(part[:, i] - part[:, j]))
        sum_dif_min += np.mean(abs_difs)
        
    return sum_dif_min / p

def calculate_gini(array) -> float:
    """
    Calculate the Gini coefficient of a numpy array
    """
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(array, array)).mean()
    # Relative mean absolute difference
    rmad = mad / np.abs(array).mean() if np.abs(array).mean() > 0 else 0
    # Gini coefficient
    gini = 0.5 * rmad
    return gini

def calculate_weights_gini(model):
    """
    Calculate Gini coefficients for weights in a PyTorch model
    Returns a dictionary with Gini coefficients for each weight matrix
    """
    W1 = model.fc1.weight.detach().cpu().numpy().flatten()
    W2 = model.fc2.weight.detach().cpu().numpy().flatten()
    
    gini_W1 = calculate_gini(W1)
    gini_W2 = calculate_gini(W2)
    
    return {
        'gini_W1': gini_W1,
        'gini_W2': gini_W2
    }

if __name__ == "__main__":
  PATH = r"\delayed_L2_13_lr_0.01_seed_1\saved_weights" #sample path, change to the directory where your model weights are stored
  p = 13
  lr = 0.01
  lam = 0.0001
  seed = 3
  proportion = 0.8
  number_of_hidden_units = int((p + 1) * p / 2)
  #set seed for reproducibility
  torch.manual_seed(seed)
  np.random.seed(seed)

  # Generate data
  inputs, labels = gen_data(p)
  sorted_inputs, sorted_labels = sorted_inputs(p)
  # Split data into train and validation sets
  train_inputs, train_labels, val_inputs, val_labels = split_train_validation(inputs, labels, proportion=proportion, seed=seed)
  print(train_inputs)

  # Get all .pt files and parse epoch numbers
  all_files = os.listdir(PATH)
  model_files_with_epochs = []
  for f_name in all_files:
      if f_name.startswith('weights_epoch_') and f_name.endswith('.pt'): # Assuming a consistent naming pattern
          try:
              # Extract epoch number from filename like "weights_epoch_1000.pt"
              epoch_num_str = f_name.split('_')[-1].split('.')[0]
              epoch_num = int(epoch_num_str)
              model_files_with_epochs.append({'epoch': epoch_num, 'filename': f_name})
          except ValueError:
              print(f"Warning: Could not parse epoch from filename: {f_name}. Skipping.")
              continue # Skip files that don't match the expected format

  # Sort files by epoch number
  model_files_with_epochs.sort(key=lambda x: x['epoch'])

  if not model_files_with_epochs:
      print(f"No model files found in {PATH} matching the pattern 'weights_epoch_*.pt'. Exiting.")
      sys.exit()

  # Initialize lists to store metrics
  epochs_collected = []
  accuracy_collected_train = []
  accuracy_collected_val = []
  complexity_collected = []
  sparsity_collected = []
  entropy_collected = []
  linf_norm_collected_W2 = []
  similarity_collected = []

  # calculate the statistics for each model in sorted order
  print(f"Found {len(model_files_with_epochs)} model files to process in sorted order.")
  for file_info in model_files_with_epochs:
      epoch = file_info['epoch']
      model_file_name = file_info['filename']
      
      print(f"Processing {model_file_name} (Epoch: {epoch})...")

      # Load the model
      model = ModuloAdditionNet(p, number_of_hidden_units=number_of_hidden_units)
      model_path = os.path.join(PATH, model_file_name)
      model = load_weights(model, model_path) # Assuming load_weights modifies model in-place or returns it

      # Make predictions on the validation set
      with torch.no_grad():
          train_outputs = model(train_inputs)
          train_predictions = torch.argmax(train_outputs, dim=1)
          val_outputs = model(val_inputs)
          val_predictions = torch.argmax(val_outputs, dim=1)
          # Ensure your metric functions return single scalar values or handle tensors appropriately
          approximate_local_circuit_complexity_value = approximate_local_circuit_complexity(model, val_inputs, seed).item() # .item() if it's a tensor
          activation_sparsity_value = activation_sparsity(train_inputs, p, number_of_hidden_units, 0.1, model)
          absolute_weight_entropy_value = absolute_weight_entropy(model)
          similarity_value = similarity(model, sorted_inputs, p)
          L_infinity_norm_value_W2= L_infty_norm(model)
      # Calculate accuracy
      accuracy_train = (train_predictions == train_labels).float().mean().item()
      accuracy_val = (val_predictions == val_labels).float().mean().item()
      
      # Store metrics
      epochs_collected.append(epoch)
      accuracy_collected_train.append(accuracy_train)
      accuracy_collected_val.append(accuracy_val)
      complexity_collected.append(approximate_local_circuit_complexity_value)
      sparsity_collected.append(activation_sparsity_value)
      entropy_collected.append(absolute_weight_entropy_value)
      linf_norm_collected_W2.append(L_infinity_norm_value_W2)
      similarity_collected.append(similarity_value)

  # Plotting the metrics
  num_metrics_to_plot = 6
  fig, axs = plt.subplots(num_metrics_to_plot, 1, figsize=(12, 4 * num_metrics_to_plot), sharex=True)
  fig.suptitle(f'Model Metrics vs. Epoch\n(p={p}, lr={lr}, lam={lam}, seed={seed}, proportion={proportion})', fontsize=16)

  plot_definitions = [
      {'data': accuracy_collected_train, 'title': 'Validation Accuracy', 'ylabel': 'Accuracy'},
      {'data': complexity_collected, 'title': 'Approx. Local Circuit Complexity (Val Inputs)', 'ylabel': 'KL Divergence'},
      {'data': sparsity_collected, 'title': 'Activation Sparsity (Train Inputs, tau=0.1)', 'ylabel': 'Sparsity'},
      {'data': entropy_collected, 'title': 'Absolute Weight Entropy (fc1)', 'ylabel': 'Entropy'},
      {'data': linf_norm_collected_W2, 'title': 'L-infinity Norm of W2', 'ylabel': 'L-inf Norm W2'},
      {'data': similarity_collected, 'title': 'Similarity Measure (Sorted Inputs)', 'ylabel': 'Similarity Score'}
  ]

  if not epochs_collected:
      print("No data collected to plot.")
  else:
      for i, pdef in enumerate(plot_definitions):
          axs[i].plot(epochs_collected, pdef['data'], marker='.', linestyle='-')
          axs[i].set_ylabel(pdef['ylabel'])
          axs[i].set_title(pdef['title'])
          axs[i].grid(True, linestyle='--', alpha=0.7)
          if i == 0:
              axs[i].plot(epochs_collected, accuracy_collected_val, marker='.', linestyle='-')

      axs[-1].set_xlabel("Epoch")
      plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for subtitle
      #save plots
      save_path = os.path.join(PATH, f"model_metrics_p_{p}_lr_{lr}_lam_{lam}_seed_{seed}.pdf")
      plt.savefig(save_path)
      plt.show()

