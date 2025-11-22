"""
Problem 1: Neural Networks (20 pts)
3-layer fully connected neural network implementation using PyTorch
"""

import torch
import torch.nn.functional as F

# Define the custom activation function: f(x) = ReLU(x) + sin(x)
def custom_activation(x):
    """
    Custom activation function: ReLU(x) + sin(x)
    """
    return torch.clamp(x, min=0) + torch.sin(x)

# Define the derivative of the custom activation function
def custom_activation_derivative(x):
    """
    Derivative of custom activation: 
    - If x > 0: 1 + cos(x)
    - If x <= 0: cos(x)
    """
    return torch.where(x > 0, 1 + torch.cos(x), torch.cos(x))

# Define softmax activation function
def softmax(z):
    """
    Softmax activation: σ(z_i) = e^(z_i) / (sum_j e^(z_j))
    """
    # Subtract max for numerical stability
    exp_z = torch.exp(z - torch.max(z))
    return exp_z / torch.sum(exp_z)

# Define cross-entropy loss function
def cross_entropy_loss(y_pred, y_true):
    """
    Cross-entropy loss: Loss = -Σ y_i * log(ŷ_i)
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = torch.clamp(y_pred, min=epsilon, max=1 - epsilon)
    return -torch.sum(y_true * torch.log(y_pred))

# Initialize network parameters
def initialize_parameters():
    """
    Initialize all network parameters as specified in the problem
    """
    # Input vector x
    x = torch.tensor([[1.0],
                      [-1.0],
                      [0.5],
                      [2.0]], dtype=torch.float32)
    
    # Weight matrix W_1 (3x4)
    W_1 = torch.tensor([[0.1, -0.2, 0.3, 0.4],
                        [0.5, -0.3, 0.1, -0.2],
                        [0.4, 0.2, -0.5, 0.3]], dtype=torch.float32)
    
    # Bias vector b_1 (3x1)
    b_1 = torch.tensor([[0.1],
                        [-0.1],
                        [0.05]], dtype=torch.float32)
    
    # Weight matrix W_2 (2x3)
    W_2 = torch.tensor([[-0.3, 0.2, 0.1],
                        [0.4, -0.5, 0.3]], dtype=torch.float32)
    
    # Bias vector b_2 (2x1)
    b_2 = torch.tensor([[0.05],
                        [-0.05]], dtype=torch.float32)
    
    return x, W_1, b_1, W_2, b_2

# Task 1: Forward pass
def forward_pass(x, W_1, b_1, W_2, b_2):
    """
    Perform forward pass through the network
    
    Returns:
        Z_1: net input to hidden layer (3x1)
        H: activation output of hidden layer (3x1)
        Z_2: net input to output layer (2x1)
        y_pred: final output after softmax (2x1)
    """
    # Forward pass through hidden layer
    # Z_1 = W_1 * x + b_1
    Z_1 = torch.matmul(W_1, x) + b_1
    
    # H = f(Z_1) = ReLU(Z_1) + sin(Z_1)
    H = custom_activation(Z_1)
    
    # Forward pass through output layer
    # Z_2 = W_2 * H + b_2
    Z_2 = torch.matmul(W_2, H) + b_2
    
    # y_pred = softmax(Z_2)
    y_pred = softmax(Z_2)
    
    return Z_1, H, Z_2, y_pred

# Task 2: Backward pass (gradient computation)
def backward_pass(x, y_true, Z_1, H, Z_2, y_pred, W_1, W_2):
    """
    Compute gradients of loss with respect to all parameters
    
    Returns:
        dW_1: gradient w.r.t. W_1
        db_1: gradient w.r.t. b_1
        dW_2: gradient w.r.t. W_2
        db_2: gradient w.r.t. b_2
    """
    # Error term for output layer: δ_2 = y_pred - y_true
    # This is the derivative of cross-entropy loss w.r.t. Z_2
    # Softmax层的梯度 = (模型预测的概率 - 真实标签的one-hot编码)
    delta_2 = y_pred - y_true
    
    # Gradient w.r.t. W_2: dW_2 = δ_2 * H^T
    dW_2 = torch.matmul(delta_2, H.T)
    
    # Gradient w.r.t. b_2: db_2 = δ_2
    db_2 = delta_2
    
    # Error term for hidden layer: δ_1 = (W_2^T * δ_2) ⊙ f'(Z_1)
    # where ⊙ denotes element-wise multiplication
    W_2_T_delta_2 = torch.matmul(W_2.T, delta_2)
    f_prime_Z_1 = custom_activation_derivative(Z_1)
    delta_1 = W_2_T_delta_2 * f_prime_Z_1
    
    # Gradient w.r.t. W_1: dW_1 = δ_1 * x^T
    dW_1 = torch.matmul(delta_1, x.T)
    
    # Gradient w.r.t. b_1: db_1 = δ_1
    db_1 = delta_1
    
    return dW_1, db_1, dW_2, db_2

# Task 3: Parameter update
def update_parameters(W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, learning_rate):
    """
    Update parameters using gradient descent
    
    Returns:
        Updated parameters
    """
    W_1_new = W_1 - learning_rate * dW_1
    b_1_new = b_1 - learning_rate * db_1
    W_2_new = W_2 - learning_rate * dW_2
    b_2_new = b_2 - learning_rate * db_2
    
    return W_1_new, b_1_new, W_2_new, b_2_new

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("Problem 1: Neural Networks Solution")
    print("=" * 80)
    
    # Initialize parameters
    x, W_1, b_1, W_2, b_2 = initialize_parameters()
    
    # For demonstration, we need a target output y
    # Since not specified, we'll use a one-hot vector [1, 0] as example
    y_true = torch.tensor([[1.0],
                           [0.0]], dtype=torch.float32)
    
    print("\n" + "=" * 80)
    print("Task 1: Forward Pass")
    print("=" * 80)
    
    # Task 1: Forward pass
    Z_1, H, Z_2, y_pred = forward_pass(x, W_1, b_1, W_2, b_2)
    
    print("\nInput vector x:")
    print(x)
    print("\nWeight matrix W_1:")
    print(W_1)
    print("\nBias vector b_1:")
    print(b_1)
    print("\nWeight matrix W_2:")
    print(W_2)
    print("\nBias vector b_2:")
    print(b_2)
    
    print("\n" + "-" * 80)
    print("Forward Pass Results:")
    print("-" * 80)
    print("\nZ_1 (net input to hidden layer):")
    print(torch.round(Z_1, decimals=4))
    print("\nH (activation output of hidden layer):")
    print(torch.round(H, decimals=4))
    print("\nZ_2 (net input to output layer):")
    print(torch.round(Z_2, decimals=4))
    print("\nŷ (final output after softmax):")
    print(torch.round(y_pred, decimals=4))
    
    # Calculate loss
    loss = cross_entropy_loss(y_pred, y_true)
    print("\nCross-entropy Loss:")
    print(torch.round(loss, decimals=4))
    
    print("\n" + "=" * 80)
    print("Task 2: Gradient Computation")
    print("=" * 80)
    
    # Task 2: Backward pass
    dW_1, db_1, dW_2, db_2 = backward_pass(x, y_true, Z_1, H, Z_2, y_pred, W_1, W_2)
    
    print("\nGradients:")
    print("\ndW_1 (gradient w.r.t. W_1):")
    print(torch.round(dW_1, decimals=4))
    print("\ndb_1 (gradient w.r.t. b_1):")
    print(torch.round(db_1, decimals=4))
    print("\ndW_2 (gradient w.r.t. W_2):")
    print(torch.round(dW_2, decimals=4))
    print("\ndb_2 (gradient w.r.t. b_2):")
    print(torch.round(db_2, decimals=4))
    
    print("\n" + "=" * 80)
    print("Task 3: Parameter Update")
    print("=" * 80)
    
    # Task 3: Parameter update
    learning_rate = 0.001
    W_1_new, b_1_new, W_2_new, b_2_new = update_parameters(
        W_1, b_1, W_2, b_2, dW_1, db_1, dW_2, db_2, learning_rate
    )
    
    print(f"\nLearning rate α = {learning_rate}")
    print("\nUpdated Parameters:")
    print("\nW_1 (updated):")
    print(torch.round(W_1_new, decimals=4))
    print("\nb_1 (updated):")
    print(torch.round(b_1_new, decimals=4))
    print("\nW_2 (updated):")
    print(torch.round(W_2_new, decimals=4))
    print("\nb_2 (updated):")
    print(torch.round(b_2_new, decimals=4))
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nAll values are rounded to 4 decimal places as required.")
    print("\nNote: The target output y_true is set to [1, 0] for demonstration.")
    print("=" * 80)

