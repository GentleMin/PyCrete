"""
Physics-Informed Neural Networks for eigenvalue problems
"""


import numpy as np
import torch
from . import pinn_core as core


class EigenNet1D(core.PhysicsInformedNet):
    """Eigenvalue-eigenfunction solver for general 1D system
    
    :param x_range: Tensor, lower and upper limits of the physical domain
    :param n_int: int, number of interior sampling points
    :param bc_weight: float, regularization strength of the boundary loss
    """
    
    def __init__(self, x_range, n_int, bc_weight=1) -> None:
        super().__init__(x_range, n_int, bc_weight)
        self.jac = (self.x_range[0, 1] - self.x_range[0, 0])/2
        self.train_pt_int = torch.atleast_2d(torch.linspace(self.x_range[0, 0], self.x_range[0, 1], self.n_int)).T
    
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
    
    def compute_loss(self):
        """Combine the losses
        """
        loss_int = self.residual(self.train_pt_int)
        loss_bc = self.boundary_loss()
        loss = loss_int + self.bc_weight*loss_bc
        return torch.log10(loss), loss_int, loss_bc


class EigenFuncNN1D(EigenNet1D):
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
        
        super().__init__(x_range=x_range, n_int=n_int, bc_weight=bc_weight)
        self.eigval = eigval
        self.mag_weight = mag_weight
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=self.x_range.shape[0])
        self.train_pt_int = self.sobol_to_physics_dom(self.sobol_engine.draw(self.n_int))
    
    def sobol_to_physics_dom(self, sobol_samples):
        """Convert Sobol points to physics domain points
        
        :param sobol_samples: Tensor, Sobol samples, within range [0, 1]
        :returns physics_samples: Tensor, Sobol samples converted to physical domain
        """
        physics_samples = sobol_samples*(self.x_range[0, 1] - self.x_range[0, 0]) + self.x_range[0, 0]
        return physics_samples
    
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


class EigenSolveNet1D(EigenNet1D):
    """Neural network eigenvalue problem solver for 1-D system
    This class builds a neural network pipeline that aims to solve the eigenvalue-eigenfunction pair simultaneously
    Abstract class - to be overridden
    
    :param x_range: Tensor, lower and upper limits of the physical domain
    :param n_int: int, number of interior sampling points
    :param bc_weight: float, regularization strength of the boundary loss
    :param mag_weight: float, regularization strength of the magnitude loss
    :param drive_weight: float, regularization strenght of driver term
    :param scan_floor: float, shift constant in driver term
    """
    
    def __init__(self, x_range=torch.tensor([[-1, 1]]), n_int=100, bc_weight=1., mag_weight=1., drive_weight=1., scan_floor=0., scan_sign=+1) -> None:
        
        super().__init__(x_range=x_range, n_int=n_int, bc_weight=bc_weight)
        self.mag_weight = mag_weight
        self.drive_weight = drive_weight
        self.scan_floor = torch.Tensor([scan_floor, ])
        self.scan_sign = scan_sign
        self.eigenvalue = torch.Tensor([scan_floor, ])
        self.eigenvalue.requires_grad = True
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=self.x_range.shape[0])
        self.train_pt_int = self.sobol_to_physics_dom(self.sobol_engine.draw(self.n_int))
    
    def sobol_to_physics_dom(self, sobol_samples):
        """Convert Sobol points to physics domain points
        
        :param sobol_samples: Tensor, Sobol samples, within range [0, 1]
        :returns physics_samples: Tensor, Sobol samples converted to physical domain
        """
        physics_samples = sobol_samples*(self.x_range[0, 1] - self.x_range[0, 0]) + self.x_range[0, 0]
        return physics_samples
    
    def boundary_loss(self):
        """Boundary loss - penalizing boundary condition discrepancy
        current implementation penalizes L2 norm of the boundary values
        effectively enforcing homogeneous Dirichlet boundary condition
        can be further overridden
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
        modulus_squared = (self.x_range[0, 1] - self.x_range[0, 0])*torch.mean(self.forward(x_int)**2)
        return (modulus_squared - 1.)**2
    
    def driver_loss(self):
        """Driver loss
        """
        penalty = torch.exp(self.scan_sign*(self.scan_floor - self.eigenvalue)).squeeze()
        return penalty
    
    def compute_loss(self):
        """Conglomerate the losses
        """
        loss_int = self.residual(self.train_pt_int)
        loss_bc = self.boundary_loss()
        loss_norm = self.magnitude_loss(self.train_pt_int)
        loss_drive = self.driver_loss()
        loss = loss_int + self.bc_weight*loss_bc + self.mag_weight*loss_norm + self.drive_weight*loss_drive
        return torch.log10(loss), loss_int, loss_bc, loss_norm, loss_drive

