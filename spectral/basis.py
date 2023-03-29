"""
Utilites for evaluating global basis functions
and evaluating functions expanded in these basis

Jingtao Min @ ETH Zurich, 2023
"""


import numpy as np
import scipy.special as specfun


class ChebyshevTSpace:
    
    def __init__(self, degrees, xcoord: np.ndarray) -> None:
        if np.array(degrees).ndim == 0:
            self.degrees = np.arange(degrees)
        else:
            self.degrees = degrees
        self.xcoord = xcoord
        Xi_mesh, N_mesh = np.meshgrid(self.xcoord, self.degrees, indexing='ij')
        self.basis = specfun.eval_chebyt(N_mesh, Xi_mesh)
    
    def __call__(self, coeffs: np.ndarray) -> np.ndarray:
        if coeffs.ndim == 1:
            assert coeffs.size == self.degrees.size
            return np.sum(coeffs*self.basis, axis=-1)
        elif coeffs.ndim == 2:
            assert coeffs.shape[1] == self.degrees.size
            return np.array([np.sum(coeffs[i]*self.basis, axis=-1) for i in range(coeffs.shape[0])])
    
    def kernel(self, integrand, int_degree=0, jacobi_idx=(-1/2, -1/2), transform=lambda x: x, jac=lambda x: np.ones(x.shape)):
        
        n_degree = 2*int(self.degrees.max()) + int_degree
        n_quad = n_degree//2 + 1
        xi_quad, wt_quad = specfun.roots_jacobi(n_quad, jacobi_idx[0], jacobi_idx[1])
        
        N_mesh, Xi_mesh = np.meshgrid(self.degrees, xi_quad, indexing='ij')
        basis_quad = specfun.eval_chebyt(N_mesh, Xi_mesh)
        int_wts = wt_quad*integrand(transform(xi_quad))*jac(xi_quad)
        kernel_matrix = np.sum(int_wts*(basis_quad[:, np.newaxis, :]*basis_quad[np.newaxis, :, :]), axis=-1)
        
        return kernel_matrix

