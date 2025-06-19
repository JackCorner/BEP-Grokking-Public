import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


from utility_functions import gen_data, split_train_validation, ModuloAdditionNet, load_weights, extract_weights, sorted_inputs, sorted_inputs_display

def apply_invariance_remove_b(W_1,W_2,b):
    """
    Apply invariance removal to the weights and biases of a neural network layer.
    
    Parameters:
    W_1 (torch.Tensor): Weights of the first layer.
    W_2 (torch.Tensor): Weights of the second layer.
    b (torch.Tensor): Biases of the second layer.
    
    Returns:
    torch.Tensor: Updated weights and biases after invariance removal.
    """
    for i in range(len(b)):
        a_i = 0.5 * b[i]
        W_1[i] = W_1[i] + a_i * np.ones(W_1.shape[1])
    
    return W_1, W_2

def apply_invariance_scalling(W_1, W_2, b=None, type='max', matrix='W_1'):
    """
    Apply invariance scaling to the weights and biases of a neural network layer.
    Scales rows of W_1 and columns of W_2.
    If matrix='W_1', the norm is derived from rows of W_1. W_1 rows are normalized, W_2 columns are scaled inversely.
    If matrix='W_2', the norm is derived from columns of W_2. W_2 columns are normalized, W_1 rows are scaled inversely.
    
    Parameters:
    W_1 (np.ndarray): Weights of the first layer.
    W_2 (np.ndarray): Weights of the second layer.
    b (np.ndarray, optional): Biases of the first layer (fc1.bias). Scales with W_1 rows.
    type (str): Type of norm ('max', 'l1', 'l2').
    matrix (str): Matrix to base normalization on ('W_1' or 'W_2').
    
    Returns:
    tuple: Updated W_1, W_2 (and b if provided).
    """
    num_hidden_units = W_1.shape[0]
    if W_2.shape[1] != num_hidden_units:
        # W_2.shape[0] is also the number of hidden units (rows in W_2)
        raise ValueError(
            f"W_1 rows ({W_1.shape[0]}) and W_2 rows ({W_2.shape[0]}) must match (number of hidden units)."
        )

    for neuron_index in range(num_hidden_units):
        current_norm = 0.0

        if matrix == 'W_1':
            # Calculate norm of the neuron_index-th row of W_1
            row_W1 = W_1[neuron_index, :]
            if type == 'max':
                current_norm = np.max(np.abs(row_W1))
            elif type == 'l2':
                current_norm = np.linalg.norm(row_W1, 2)
            elif type == 'l1':
                current_norm = np.linalg.norm(row_W1, 1)
            else:
                raise ValueError("Invalid type for norm. Choose 'max', 'l1', or 'l2'.")
        elif matrix == 'W_2':
            # Calculate norm of the neuron_index-th column of W_2
            col_W2 = W_2[:, neuron_index]
            if type == 'max':
                current_norm = np.max(np.abs(col_W2))
            elif type == 'l2':
                current_norm = np.linalg.norm(col_W2, 2)
            elif type == 'l1':
                current_norm = np.linalg.norm(col_W2, 1)
            else:
                raise ValueError("Invalid type for norm. Choose 'max', 'l1', or 'l2'.")
        else:
            raise ValueError("Invalid matrix for norm base. Choose 'W_1' or 'W_2'.")

        if current_norm == 0:
            continue 

        factor_for_W1_row = 1.0
        factor_for_W2_col = 1.0

        if matrix == 'W_1':
            factor_for_W1_row = 1.0 / current_norm
            factor_for_W2_col = current_norm
        elif matrix == 'W_2': # matrix == 'W_2'
            factor_for_W2_col = 1.0 / current_norm
            factor_for_W1_row = current_norm
        
        # Apply scaling
        W_1[neuron_index, :] *= factor_for_W1_row
        W_2[:, neuron_index] *= factor_for_W2_col # Scale the neuron_index-th column of W_2
        
        if b is not None:
            b[neuron_index] *= factor_for_W1_row
            
    if b is not None:
        return W_1, W_2, b
    else:
        return W_1, W_2
        

def generate_heatmaps(W_1_orig, W_2_orig, b_1_orig, p, lr, lam, epoch, seed, save_directory, type='max', matrix='W_1', height=8, width=10):
    """
    Generate heatmaps for the weights and biases of a neural network layer after applying invariances.
    
    Parameters:
    W_1_orig (np.ndarray): Original weights of the first layer.
    W_2_orig (np.ndarray): Original weights of the second layer.
    b_1_orig (np.ndarray): Original biases of the first layer.
    p (int): Modulo value.
    save_directory (str): Directory to save the generated heatmaps.
    type (str): Type of norm for scaling ('max', 'l1', 'l2').
    matrix (str): Matrix to base scaling on ('W_1' or 'W_2').
    
    Returns:
    None
    """
    os.makedirs(save_directory, exist_ok=True)
    # Apply invariances to copies of the weights
    W_1 = W_1_orig.copy()
    W_2 = W_2_orig.copy()
    b_1 = b_1_orig.copy()

    W_1, W_2 = apply_invariance_remove_b(W_1, W_2, b_1) # b_1 is fc1.bias, W_1 is modified

    # apply_invariance_scalling modifies W_1, W_2, and b_1 in place if b_1 is passed
    if b_1 is not None:
        W_1, W_2, b_1 = apply_invariance_scalling(W_1, W_2, b_1, type=type, matrix=matrix)
    else:
        W_1, W_2 = apply_invariance_scalling(W_1, W_2, None, type=type, matrix=matrix)
    


    # Sorted inputs for Ax plots
    s_inputs, _ = sorted_inputs(p) # From utility_functions
    s_inputs_np = s_inputs.cpu().numpy() # Convert to NumPy
    sorted_inputs_for_display = sorted_inputs_display(p)

    output_first_layer = W_1 @ s_inputs_np.T
    relu_output_first_layer = np.maximum(output_first_layer, 0)

    # --- Heatmap of Ax output ---
    fig, ax = plt.subplots(figsize=(width+2, height)) 
    #make the text size bigger
    plt.rcParams.update({'font.size': 16})
    im = ax.imshow(output_first_layer, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im)
    plt.title(f"Pre-activation Output ($Ax$)")
    plt.ylabel("Neuron index")
    ax.set_xticks(np.arange(len(s_inputs_np)))
    ax.set_xticklabels(sorted_inputs_for_display, rotation=45, ha='right')
    ax.set_xlabel("Input vector $x$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"heatmap_W1x_p_{p}_seed_{seed}lr_{lr}_lam_{lam}_epoch_{epoch}_{type}_{matrix}.pdf"))
    # plt.show()
    plt.close(fig)

    # --- Heatmap of ReLU(Ax) output ---
    fig, ax = plt.subplots(figsize=(width+2, height)) 
    #make the text size bigger
    plt.rcParams.update({'font.size': 16})
    im = ax.imshow(relu_output_first_layer, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im)
    plt.title(f"Activation Output (ReLU($Ax$))")
    plt.ylabel("Neuron index")
    ax.set_xticks(np.arange(len(s_inputs_np)))
    ax.set_xticklabels(sorted_inputs_for_display, rotation=45, ha='right')
    ax.set_xlabel("Input vector $x$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"heatmap_ReLU_W1x_p_{p}_seed_{seed}lr_{lr}_lam_{lam}_epoch_{epoch}_{type}_{matrix}.pdf"))
    # plt.show()
    plt.close(fig)

    # --- Heatmap of W_1 ---
    x_labels = []
    for num1 in range(2*p):
        pos = "second" if num1 >= p else "first"
        x_labels.append(f"{num1%p}")

    fig, ax = plt.subplots(figsize=(width, height))
    #make the text size bigger
    plt.rcParams.update({'font.size': 16})
    im = ax.imshow(W_1, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im)
    plt.title(f"Matrix $A$")
    plt.ylabel("Neuron index")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel("Input feature index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"heatmap_W1_p_{p}_seed_{seed}lr_{lr}_lam_{lam}_epoch_{epoch}_{type}_{matrix}.pdf"))
    # plt.show()
    plt.close(fig)

    # --- Heatmap of ReLU(W_1) ---
    relu_w_1 = np.maximum(W_1, 0)
    fig, ax = plt.subplots(figsize=(width, height))
    #make the text size bigger
    plt.rcParams.update({'font.size': 16})
    im = ax.imshow(relu_w_1, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im)
    plt.title(f"ReLU($A$)")
    plt.ylabel("Neuron index")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_xlabel("Input feature index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, f"heatmap_ReLU_W1_p_{p}_seed_{seed}lr_{lr}_lam_{lam}_epoch_{epoch}_{type}_{matrix}.pdf"))
    plt.close(fig)
    
    print(f"Heatmaps saved to {save_directory}")


if __name__ == "__main__":

    # relative path to the model file
    model_relative_path = r"modulo_addition_p_3_lr_0.01_lam_5e-05_seed_4_epoch_300000.pt"

    # parameters used for naming the heatmap files (just for book keeping)
    seed = 4	
    epoch=300_000
    lr = 0.01
    lam = 0.00005

    # training parameters
    p = 3 # modulo value
    number_of_hidden_units = int((p + 1) * p / 2) # number of hidden units in the hidden layer

    invariance_norm_type = 'max' # Type of norm to use for invariance scaling ('max', 'l1', 'l2')
    invariance_scaling_matrix = 'W_1' # Normalize rows of W_1, scale columns of W_2

    

    ### code to generate heatmaps
    heatmap_save_dir = os.path.join("\generated_heatmaps")
    

    net = ModuloAdditionNet(p, number_of_hidden_units=number_of_hidden_units)
    
    print(f"Loading model from: {model_relative_path}")
    if not os.path.exists(model_relative_path):
        print(f"ERROR: Model file not found at {model_relative_path}")
        print("Please check the path to your pre-trained model.")
    else:
        # load the model weights
        net = load_weights(net, model_relative_path)
        print("Model loaded successfully.")

        # Extract weights using utility_functions.extract_weights
        W1_numpy, b1_numpy, W2_numpy = extract_weights(net)

        # Generate heatmaps
        print(f"Generating heatmaps with p={p}, norm_type='{invariance_norm_type}', scaling_matrix='{invariance_scaling_matrix}'")
        generate_heatmaps(
            W_1_orig=W1_numpy, 
            W_2_orig=W2_numpy, 
            b_1_orig=b1_numpy, 
            p=p,
            lr=lr,
            lam=lam,
            epoch=epoch,
            seed=seed, 
            save_directory=heatmap_save_dir,
            type=invariance_norm_type,
            matrix=invariance_scaling_matrix
        )



