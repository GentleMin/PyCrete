"""
Physics-Informed Neural Networks for eigenvalue problems
"""


import numpy as np
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


class EigenFuncNN1D:
    """Neural network eigenfunction solver for 1-D system
    This class builds a neural network that aims to solve the eigenfunction given an eigenvalue to the system
    Abstract class - to be overridden
    
    :param eigval: float, eigenvalue
    :param x_range: Tensor, lower and upper limits of the physical domain
    :param n_int: int, number of interior sampling points
    :param bc_weight: float, regularization strength of the boundary loss
    :param mag_weight: float, regularization strength of the magnitude loss
    """
    
    def __init__(self, eigval, x_range=torch.tensor([[-1, 1]]), n_int=100, bc_weight=1., mag_weight=1.) -> None:
        
        self.eigval = eigval
        self.x_range = x_range
        self.n_int = n_int
        self.bc_weight = bc_weight
        self.mag_weight = mag_weight
        self.jac = (self.x_range[0, 1] - self.x_range[0, 0])/2
        
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=self.x_range.shape[0])
        self.train_pt_int = self.sobol_to_physics_dom(self.sobol_engine.draw(self.n_int))
    
    def sobol_to_physics_dom(self, sobol_samples):
        """Convert Sobol points to physics domain points
        
        :param sobol_samples: Tensor, Sobol samples, within range [0, 1]
        :returns physics_samples: Tensor, Sobol samples converted to physical domain
        """
        physics_samples = sobol_samples*(self.x_range[0, 1] - self.x_range[0, 0]) + self.x_range[0, 0]
        return physics_samples
    
    def normalize(self, x_input):
        """Normalize physical domain points to [-1, 1]
        
        :param x_input: Tensor[batch_size, 1], input points in physical domain
        :returns x_normalized: Tensor of the same shape, in [-1, 1]
        """
        x_normalized = (x_input - self.x_range[0, 0])/self.jac - 1
        return x_normalized
    
    def denormalize(self, x_input):
        """Denormalize points in [-1, 1] back to the physical domain
        
        :param x_input: Tensor[batch_size, 1], input points in [-1, 1]
        :returns x_normalized: Tensor of the same shape, in physical domain
        """
        x_denormalized = (x_input + 1)*self.jac + self.x_range[0, 0]
        return x_denormalized
    
    def forward(self, x_input):
        """This implements the feed-forward pass
        Intended use: construct a torch module network to use as the forward operator
        and call the network to conduct the forward pass
        
        :param x_input: Tensor[batch_size, 1], input points in physical domain
        """
        raise NotImplementedError
    
    def residual(self, x_int):
        """This implements the PDE loss
        The implementation will be problem-dependent
        
        :param x_int: Tensor[batch_size, 1], interior points in physical domain
        """
        raise NotImplementedError
    
    def boundary_loss(self):
        """Boundary loss - penalizing boundary condition discrepancy
        current implementation penalizes L2 norm of the boundary values
        effectively enforcing homogeneous Dirichlet boundary condition
        can be overridden
        """
        val_bc = self.forward(self.x_range.T)
        val_train = torch.zeros(self.x_range.T.shape)
        loss_bc = torch.mean(torch.abs(val_bc - val_train)**2)
        return loss_bc
    
    def magnitude_loss(self, x_int):
        """Magntitude loss - penalizing non-unity magntitude
        current implementation penalizes L2 misfit between the magnitude and 1
        can be overridden
        
        :param x_int: Tensor[batch_size, 1], interior points in physical domain
        """
        modulus_squared = torch.mean(self.forward(x_int)**2)
        return (modulus_squared - 1.)**2
    
    def compute_loss(self):
        """Conglomerate the losses
        """
        loss_int = self.residual(self.train_pt_int)
        loss_bc = self.boundary_loss()
        loss_mod = self.magnitude_loss(self.train_pt_int)
        loss = loss_int + self.bc_weight*loss_bc + self.mag_weight*loss_mod
        return torch.log10(loss), loss_int, loss_bc, loss_mod
    