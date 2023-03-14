"""
Utilites for numerical integration / quadrature

Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np


quad_1d = {
    "1-pt": [np.array([0]), 
             np.array([2])],
    "2-pt": [np.array([-1, 1])/np.sqrt(3), 
             np.array([1, 1])],
    "3-pt": [np.array([-1, 0, +1])*np.sqrt(3/5), 
             np.array([5, 8, 5])/9],
}


quad_2d = {
    "1-pt": [np.array([[0],
                       [0]]), 
             np.array([4])],
    "4-pt": [np.array([[-1, -1, 1, 1], 
                       [-1, 1, 1, -1]])/np.sqrt(3), 
             np.array([1, 1, 1, 1])], 
    "9-pt": [np.array([[-1, -1, +1, +1, -1, 0, +1, 0, 0], 
                       [-1, +1, +1, -1, 0, +1, 0, -1, 0]])*np.sqrt(3/5), 
             np.array([25, 25, 25, 25, 40, 40, 40, 40, 64])/81]
}
