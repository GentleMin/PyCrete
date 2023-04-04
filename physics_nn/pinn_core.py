"""
Core utilities, abstract classes for Physics-informed Neural Network (PINN)
"""


import torch


def linear_init_xavier(m):
    """Use Xavier uniform initialization on linear layers.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("tanh"))

def uniform_MLP(in_dim, out_dim, neurons, n_hidden_layers, activation=torch.nn.Tanh):
    """Construct uniform multi-layer perceptron
    """
    nn_seq = list()
    nn_seq.extend([torch.nn.Linear(in_dim, neurons), activation()])
    for i_layer in range(n_hidden_layers):
        nn_seq.extend([torch.nn.Linear(neurons, neurons), activation()])
    nn_seq.append(torch.nn.Linear(neurons, out_dim))
    return torch.nn.Sequential(*nn_seq)


class PhysicsInformedNet:
    """Abstract base class for PINN
    """
    
    def __init__(self, x_range, n_int, bc_weight=1.) -> None:
        
        self.x_range = x_range
        self.n_int = n_int
        self.bc_weight = bc_weight
    
    def forward(self, x_input):
        """The feed-forward pass
        """
        raise NotImplementedError
    
    def residual(self, x_int):
        """The PDE loss / residual loss
        """
        raise NotImplementedError
    
    def boundary_loss(self):
        """The boundary loss
        """
        raise NotImplementedError
    
    def compute_loss(self):
        """Combine different losses
        """
        raise NotImplementedError

