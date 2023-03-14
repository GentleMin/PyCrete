"""
Utilites for meshing and elements

Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np


class FiniteElement:
    """Base class for finite element
    """
    def __init__(self) -> None:
        self.n_nodes = None
        self.order = None
    
    def f_shape(self, *args):
        """Evaluate shape functions
        
        axis 0: index of shape function
        axis 1,...: index of input coordinates
        """
        raise NotImplementedError
    
    def f_dshape(self, *args):
        """Evaluate derivative of shape functions
        with respect to std domain coordinates.
        
        axis 0: index of shape function
        axis 1: index of differential coordinates
        axis 2,...: index of input coordinates
        """
        raise NotImplementedError
    

class LinearElement(FiniteElement):
    """1-D linear element with 2 nodes
    
    Shape indexing / node correspondence:
    0: xi = -1
    1: xi = +1
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_nodes = 2
        self.order = 2
    
    def f_shape(self, xi):
        N_1 = (1 - np.atleast_1d(xi))/2
        N_2 = (1 + np.atleast_1d(xi))/2
        return np.stack([N_1, N_2], axis=0)
    
    def f_dshape(self, xi):
        dN_1 = -1/2*np.ones(np.atleast_1d(xi).shape)
        dN_2 = +1/2*np.ones(np.atleast_1d(xi).shape)
        return np.stack([dN_1, dN_2], axis=0)


class QuadraticElement(FiniteElement):
    """1-D quadratic element with 3 nodes
    
    Shape indexing / node correspondence:
    0: xi = -1
    1: xi = 0
    2: xi = +1
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_nodes = 3
        self.order = 3
    
    def f_shape(self, xi):
        xi = np.atleast_1d(xi)
        N_1 = xi*(xi - 1)/2
        N_2 = 1 - xi*xi
        N_3 = xi*(xi + 1)/2
        return np.stack([N_1, N_2, N_3], axis=0)

    def f_dshape(self, xi):
        xi = np.atleast_1d(xi)
        dN_1 = xi - 1/2
        dN_2 = -2*xi
        dN_3 = xi + 1/2
        return np.stack([dN_1, dN_2, dN_3], axis=0)


class BiLinearElement(FiniteElement):
    """2D Bi-linear element with 4 nodes
    
    Shape indexing / node correspondence:
             |
        1---------2
        |    |    |
    ----|----|----|-----
        |    |    |
        0---------3
             |
    0: xi = (-1, -1)
    1: xi = (-1, +1)
    2: xi = (+1, +1)
    3: xi = (+1, -1)
    """
    def __init__(self) -> None:
        super().__init__()
        self.element_1d = LinearElement()
        self.n_nodes = 4
        self.order = 2
    
    def f_shape(self, xi):
        N_dim1 = self.element_1d.f_shape(xi[0])
        N_dim2 = self.element_1d.f_shape(xi[1])
        return np.stack([N_dim1[0]*N_dim2[0], 
                         N_dim1[0]*N_dim2[1], 
                         N_dim1[1]*N_dim2[1], 
                         N_dim1[1]*N_dim2[0]], axis=0)
        # N_1 = (1 - xi[0])*(1 - xi[1])/4
        # N_2 = (1 - xi[0])*(1 + xi[1])/4
        # N_3 = (1 + xi[0])*(1 + xi[1])/4
        # N_4 = (1 + xi[0])*(1 - xi[1])/4
        # return np.stack([N_1, N_2, N_3, N_4], axis=0)
    
    def f_dshape(self, xi):
        N_dim1 = self.element_1d.f_shape(xi[0])
        N_dim2 = self.element_1d.f_shape(xi[1])
        dN_dim1 = self.element_1d.f_dshape(xi[0])
        dN_dim2 = self.element_1d.f_dshape(xi[1])
        return np.array([[dN_dim1[0]*N_dim2[0], N_dim1[0]*dN_dim2[0]], 
                         [dN_dim1[0]*N_dim2[1], N_dim1[0]*dN_dim2[1]], 
                         [dN_dim1[1]*N_dim2[1], N_dim1[1]*dN_dim2[1]], 
                         [dN_dim1[1]*N_dim2[0], N_dim1[1]*dN_dim2[0]]])
        # dN_1 = np.stack([-(1 - xi[1])/4, -(1 - xi[0])/4], axis=0)
        # dN_2 = np.stack([-(1 + xi[1])/4, +(1 - xi[0])/4], axis=0)
        # dN_3 = np.stack([+(1 + xi[1])/4, +(1 + xi[0])/4], axis=0)
        # dN_4 = np.stack([+(1 - xi[1])/4, -(1 + xi[0])/4], axis=0)
        # return np.stack([dN_1, dN_2, dN_3, dN_4], axis=0)


class BiQuadraticElement(FiniteElement):
    """2-D Bi-quadratic element with 9 nodes
    
    Shape indexing / node correspondence:
             |
        1----5----2
        |    |    |
    ----4----8----6-----
        |    |    |
        0----7----3
             |
    0: xi = (-1, -1)
    1: xi = (-1, +1)
    2: xi = (+1, +1)
    3: xi = (+1, -1)
    4: xi = (-1,  0)
    5: xi = ( 0, +1)
    6: xi = (+1,  0)
    7: xi = ( 0, -1)
    8: xi = ( 0,  0)
    """
    def __init__(self) -> None:
        super().__init__()
        self.element_1d = QuadraticElement()
        self.n_nodes = 9
        self.order = 3
    
    def f_shape(self, xi):
        N_dim1 = self.element_1d.f_shape(xi[0])
        N_dim2 = self.element_1d.f_shape(xi[1])
        return np.stack([N_dim1[0]*N_dim2[0], 
                         N_dim1[0]*N_dim2[2], 
                         N_dim1[2]*N_dim2[2], 
                         N_dim1[2]*N_dim2[0], 
                         N_dim1[0]*N_dim2[1], 
                         N_dim1[1]*N_dim2[2], 
                         N_dim1[2]*N_dim2[1], 
                         N_dim1[1]*N_dim2[0], 
                         N_dim1[1]*N_dim2[1]], axis=0)
    
    def f_dshape(self, xi):
        N_dim1 = self.element_1d.f_shape(xi[0])
        N_dim2 = self.element_1d.f_shape(xi[1])
        dN_dim1 = self.element_1d.f_dshape(xi[0])
        dN_dim2 = self.element_1d.f_dshape(xi[1])
        return np.array([[dN_dim1[0]*N_dim2[0], N_dim1[0]*dN_dim2[0]], 
                         [dN_dim1[0]*N_dim2[2], N_dim1[0]*dN_dim2[2]], 
                         [dN_dim1[2]*N_dim2[2], N_dim1[2]*dN_dim2[2]], 
                         [dN_dim1[2]*N_dim2[0], N_dim1[2]*dN_dim2[0]], 
                         [dN_dim1[0]*N_dim2[1], N_dim1[0]*dN_dim2[1]], 
                         [dN_dim1[1]*N_dim2[2], N_dim1[1]*dN_dim2[2]], 
                         [dN_dim1[2]*N_dim2[1], N_dim1[2]*dN_dim2[1]], 
                         [dN_dim1[1]*N_dim2[0], N_dim1[1]*dN_dim2[0]], 
                         [dN_dim1[1]*N_dim2[1], N_dim1[1]*dN_dim2[1]]])
