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
        assert coeffs.size == self.basis.shape[1]
        return np.sum(coeffs*self.basis, axis=-1)
    
    
