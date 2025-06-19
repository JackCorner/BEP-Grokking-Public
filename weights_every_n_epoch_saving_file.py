import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


from utility_functions import gen_data, split_train_validation, ModuloAdditionNet, load_weights


#training function  
def train_model(net, train_inp, train_lbl, val_inp, val_lbl, num_classes, epochs=2000, lr=0.01, lam=1e-5, print_every=500, patience=100_000, seed=42, save_weights_every=100, save_weights_path="experiment_name", starting_epoch=0, accuracy_threshold=0.999):
    """
    Train the model with early stopping and save weights every n epochs.
    Args:
        num_classes (int): The number of output classes (e.g., p for modulo p).
        starting_epoch (int): The epoch number to start from (for resumed training).
    """

    #create the directory to save weights if it doesn't exist
    actual_weights_storage_dir = os.path.join(save_weights_path, "saved_weights")
    os.makedirs(actual_weights_storage_dir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    criterion = nn.MSELoss() #mse loss

    # Pre-allocate arrays for metrics for the current training session
    train_loss_list = np.zeros(epochs)
    train_acc_list = np.zeros(epochs)
    val_loss_list = np.zeros(epochs)
    val_acc_list = np.zeros(epochs)
    l2_loss_list = np.zeros(epochs)
    

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    accuracy_epoch = 0
    accuracy_epoch_threshold = num_classes * 30_000
    
    train_one_hot = F.one_hot(train_lbl, num_classes=num_classes).float() #one hot encode the training labels
    val_one_hot = F.one_hot(val_lbl, num_classes=num_classes).float() #one hot encode the validation labels

    epochs_completed_this_run = 0

    for epoch_idx in range(epochs): # Iterates from 0 to epochs-1
        current_total_epoch = starting_epoch + epoch_idx + 1 # Actual epoch number (1-based)
        epochs_completed_this_run = epoch_idx + 1
        
        net.train()
        logits = net(train_inp) #get logits by passing training data through the network
        
        mse_loss = criterion(logits, train_one_hot) #get MSE loss
        current_l2_loss_val = sum(torch.sum(param.pow(2)) for param in net.parameters()) #get L2 loss
        loss = mse_loss + lam * current_l2_loss_val #total loss

        net.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in net.parameters():
                param -= lr * param.grad #update the parameters using gradient descent

        preds = logits.argmax(dim=1)
        acc = (preds == train_lbl).float().mean().item() #compute accuracy


        #store train metrics
        train_loss_list[epoch_idx] = mse_loss.item()
        train_acc_list[epoch_idx] = acc
        l2_loss_list[epoch_idx] = current_l2_loss_val.item()

        # Validation Phase
        net.eval()
        with torch.no_grad():
            val_logits = net(val_inp)

            val_mse_loss = criterion(val_logits, val_one_hot)
            # Use the L2 loss calculated for the current state of weights
            val_total_loss = val_mse_loss + lam * current_l2_loss_val 

            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_lbl).float().mean().item()

            # store validation metrics
            val_loss_list[epoch_idx] = val_mse_loss.item()
            val_acc_list[epoch_idx] = val_acc


            # Early stopping check
            if val_total_loss.item() < best_val_loss:
                best_val_loss = val_total_loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve +=1
            
            # Check if we should stop early
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at total epoch {current_total_epoch}. No improvement in validation loss for {patience} epochs.")
                break

        # Check if the accuracy threshold is reached
        if val_acc >= accuracy_threshold and accuracy_epoch >= accuracy_epoch_threshold:
            print(f"Accuracy threshold reached at total epoch {current_total_epoch}. Stopping training.")
            break
        elif val_acc >= accuracy_threshold:
            accuracy_epoch += 1
        
        # Save weights every 'save_weights_every' epochs based on total epoch number
        if current_total_epoch % save_weights_every == 0:
            weights_filename = f"weights_epoch_{current_total_epoch}.pt"
            weights_filepath = os.path.join(actual_weights_storage_dir, weights_filename)
            torch.save(net.state_dict(), weights_filepath)
            print(f"Saved weights to {weights_filepath}")

        if current_total_epoch % print_every == 0:
            l2_status = "ON" if lam > 0 else "OFF"
            print(f"Epoch {current_total_epoch}/{starting_epoch + epochs} | L2 {l2_status} | Train Loss={loss.item():.5f}, Train Acc={acc*100:.2f}% "
                  f"| Val Loss={val_total_loss.item():.5f}, Val Acc={val_acc*100:.2f}%", end="")
            print()
                
    # Trim the arrays to the actual number of epochs run in this session
    train_loss_list = train_loss_list[:epochs_completed_this_run]
    train_acc_list = train_acc_list[:epochs_completed_this_run]
    val_loss_list = val_loss_list[:epochs_completed_this_run]
    val_acc_list = val_acc_list[:epochs_completed_this_run]
    l2_loss_list = l2_loss_list[:epochs_completed_this_run]

    # Regular stats
    stats = {
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_loss': val_loss_list,
        'val_acc': val_acc_list,
        'l2_loss': l2_loss_list,
        'epochs_completed_in_session': epochs_completed_this_run, 
        'last_total_epoch_recorded': starting_epoch + epochs_completed_this_run
    }
    
    return stats, net


def plot_metrics(stats, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
      stats: Dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
      save_path: If provided, save the plots to this path.
    """
    epochs = np.arange(1, len(stats['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    #set fontsize
    plt.rcParams.update({'font.size': 20})
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, stats['train_loss'], label='Train Loss')
    plt.plot(epochs, stats['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, stats['train_acc'], label='Train Accuracy')
    plt.plot(epochs, stats['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    #### parameters for the experiment
    p = 13
    number_of_hidden_units = int((p + 1) * p / 2)
    seed = 2
    lam = 1e-4
    lr = 0.01
    epochs = 2_000_000
    print_every = 1000
    patience = 2_000_000
    save_weights_every = 1000
    save_weights_path = f"saved_modulo_addition_p_{p}_lr_{lr}_lam_{lam}_seed_{seed}"


    # Generate data"
    inputs, labels = gen_data(p)
    train_inputs, train_labels, val_inputs, val_labels = split_train_validation(inputs, labels, proportion=0.8, seed=seed)

    print(train_inputs)

    # Initialize the model
    net = ModuloAdditionNet(p, number_of_hidden_units=number_of_hidden_units)

    # Train the model
    stats, trained_net = train_model(net, train_inputs, train_labels, val_inputs, val_labels,
                                     num_classes=p, epochs=epochs, lr=lr, lam=lam, print_every=print_every,
                                     patience=patience, seed=seed, save_weights_every=save_weights_every,
                                     save_weights_path=save_weights_path, starting_epoch=0)


    # epochs run
    epochs_run = stats['epochs_completed_in_session']
    # Save the final model
    final_epoch_number = stats['last_total_epoch_recorded']
    torch.save(trained_net.state_dict(), f"{save_weights_path}/modulo_addition_p_{p}_lr_{lr}_lam_{lam}_seed_{seed}_weights_epoch_{final_epoch_number}.pt")
    print(f"Final model weights saved with name: {save_weights_path}/modulo_addition_p_{p}_lr_{lr}_lam_{lam}_seed_{seed}_weights_epoch_{final_epoch_number}.pt")

    # Plot metrics
    plot_metrics(stats, save_path=f"{save_weights_path}//metrics_plot.pdf")

