# Grokking in Modular Addition

This project investigates the phenomenon of "grokking" (a delayed generalization) in a simple neural network trained to perform modular addition. The repository contains scripts to train the model, save its state over time, and analyze the evolution of its internal representations and performance metrics.

## Project Workflow

1.  **Training**: Use `weights_every_n_epoch_saving_file.py` to train a `ModuloAdditionNet` on the modular addition task. This script will periodically save the model's weights (checkpoints) to a specified directory.
2.  **Analysis**: Use `model_from_folder_loader.py` to load the series of saved checkpoints from a training run. This script calculates various metrics (e.g., accuracy, sparsity, weight norms, similarity) for each checkpoint.
3.  **Visualization**: The analysis script then plots these metrics against the training epoch, allowing for the visualization of how the model's internal structure changes as it transitions from memorization to generalization. The `invariance_and_analysis_functions.py` script can also be used to generate detailed heatmaps of weights and activations for specific checkpoints.

## File Descriptions

### `utility_functions.py`

This file contains core helper functions and the neural network definition used across the project.

-   **`gen_data(p)`**: Generates the one-hot encoded input pairs and corresponding labels for the modular addition task `(a + b) mod p`.
-   **`split_train_validation(...)`**: Splits the generated data into training and validation sets.
-   **`sorted_inputs(p)`**: Returns inputs sorted by their modular sum, which is useful for structured analysis and visualization.
-   **`ModuloAdditionNet`**: Defines the simple two-layer MLP (Multi-Layer Perceptron) used for the task.
-   **`load_weights(...)` & `extract_weights(...)`**: Utility functions for loading a model's state from a file and extracting its weight/bias parameters as NumPy arrays.

### `weights_every_n_epoch_saving_file.py`

This is the main script for training the model.

-   **`train_model(...)`**: Implements the main training loop, including gradient descent, loss calculation, and accuracy computation.
-   **Functionality**:
    -   Trains the `ModuloAdditionNet` model.
    -   Saves model checkpoints (the weights) every `n` epochs.
    -   Implements early stopping and tracks training/validation metrics.
    -   Plots the final loss and accuracy curves at the end of training.

### `invariance_and_analysis_functions.py`

This module provides functions for analyzing the learned representations by applying mathematical invariances and generating visualizations.

-   **`apply_invariance_remove_b(...)`**: Applies a theoretical invariance to the weights, effectively absorbing the bias term into the first layer's weights.
-   **`apply_invariance_scalling(...)`**: Applies a scaling invariance, normalizing the rows of the first weight matrix (`W_1`) and scaling the columns of the second (`W_2`) accordingly. This helps in comparing different neurons' structures.
-   **`generate_heatmaps(...)`**: Generates and saves a series of heatmaps to visualize the transformed weights and the network's activations on sorted inputs. This is key to understanding the internal mechanisms the network has learned.

### `model_from_folder_loader.py`

This is the main script for post-training analysis. It tracks how the model's properties evolve over time.

-   **Functionality**:
    -   Loads a sequence of model checkpoints saved during a training run.
    -   For each checkpoint, it calculates a suite of metrics designed to measure the model's complexity and structure.
-   **Metrics Calculated**:
    -   **Validation Accuracy**: To identify the point of grokking.
    -   **Activation Sparsity**: Measures how many neurons are inactive on average.
    -   **Weight Entropy**: A measure of the weight distribution's uniformity.
    -   **Approximate Local Circuit Complexity**: Measures how much the output changes when weights are perturbed.
    -   **L-infinity Norm**: The maximum absolute value in the weight matrices after normalization.
    -   **Similarity**: A custom measure (Definition 5.20) to quantify the structural similarity of neuron activations for inputs that sum to the same value.
-   **Output**: Generates and saves plots showing the evolution of each of these metrics over the training