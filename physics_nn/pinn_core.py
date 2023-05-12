"""
Core utilities, abstract classes for Physics-informed Neural Network (PINN)
"""


import torch


def linear_init_xavier(m):
    """Use Xavier uniform initialization on linear layers.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("tanh"))
        m.bias.data.fill_(0)

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
    
    def __init__(self, domain, n_int, bc_weight=1.) -> None:
        
        self.domain = domain
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


class PiNN_Cartesian(PhysicsInformedNet):
    """
    Abstract base class for PiNN 
    for solving PDE in a Cartesian box
    """
    
    def __init__(self, domain, n_int, n_bound, idx_bound, bc_weight=1, device=None) -> None:
        
        super().__init__(domain, n_int, bc_weight)
        self.n_bound = n_bound
        self.idx_bound = idx_bound
        self.device = device
        
        assert self.domain.shape[0] >= 2
        assert self.domain.shape[1] == 2
        assert len(self.n_bound) == self.domain.shape[0]
        assert len(self.idx_bound) == self.domain.shape[0]
        self.ndim = self.domain.shape[0]
        self.ndim_space = self.ndim - 1
        self.jac = (self.domain[:, 1] - self.domain[:, 0])/2
        
        self.sobol_engine = torch.quasirandom.SobolEngine(dimension=self.domain.shape[0])
        self.data_interior = self.sample_interior()
        self.data_boundary = self.sample_boundary()
    
    def to_physics_domain(self, samples):
        """Convert quasirandom samples between (0, 1) to physics domain points
        """
        assert samples.shape[1] == self.ndim
        physics_samples = samples*(self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]
        return physics_samples
    
    def project_to_boundary(self, samples, dim_idx, boundary_idx):
        """Project samples in the physical domain to boundary
        """
        assert samples.shape[1] == self.ndim
        boundary_samples = samples.detach().clone()
        boundary_samples[:, dim_idx] = torch.full(samples[:, 1].shape, self.domain[dim_idx, boundary_idx])
        return boundary_samples
    
    def normalize(self, samples):
        assert samples.shape[1] == self.ndim
        normed_samples = (samples - self.domain[:, 0])/self.jac - 1
        return normed_samples
    
    def denormalize(self, samples):
        assert samples.shape[1] == self.ndim
        denormed_samples = (samples + 1)*self.jac + self.domain[:, 0]
        return denormed_samples
    
    def sample_interior(self):
        samples_int = self.to_physics_domain(self.sobol_engine.draw(self.n_int).to(self.device))
        return samples_int
    
    def sample_boundary(self):
        samples_bound = list()
        for idim in range(self.ndim):
            samples_temp = [self.project_to_boundary(
                                self.to_physics_domain(
                                    self.sobol_engine.draw(self.n_bound[idim]).to(self.device)
                                ), idim, ibound
                            ) for ibound in self.idx_bound[idim]]
            samples_bound.append(samples_temp)
        return samples_bound
                
