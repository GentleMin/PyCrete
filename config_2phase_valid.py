"""
Physical configuration for two-phase heat equation
"""

import numpy as np

diffuse_f = 0.005
diffuse_s = 0.005
couple_f = 0.
couple_s = 0.
background_flow = 1.


"""Introducing coefficients"""

def get_background_flow(t):
    cycle_t = t - 4*np.floor(t/4)
    if cycle_t >= 0 and cycle_t <= 1:
        return 1.
    elif cycle_t >= 2 and cycle_t <= 3:
        return -1.
    else:
        return 0.

def m_u(s):
    return np.ones(s.shape)

def k_uu2(s):
    return diffuse_f*np.ones(s.shape)

def k_uu1(s):
    return -np.ones(s.shape)

def k_uu0(s):
    return -couple_f*np.ones(s.shape)

def k_ub1(s):
    return np.zeros(s.shape)
    
def k_ub0(s):
    return couple_f*np.ones(s.shape)

def m_b(s):
    return np.ones(s.shape)

def k_bu1(s):
    return np.zeros(s.shape)

def k_bu0(s):
    return couple_s*np.ones(s.shape)

def k_bb2(s):
    return diffuse_s*np.ones(s.shape)

def k_bb1(s):
    return np.zeros(s.shape)

def k_bb0(s):
    return -couple_s*np.ones(s.shape)

# Necessary derivatives

def dk_uu2(s):
    return np.zeros(s.shape)

def dk_bb2(s):
    return np.zeros(s.shape)


"""Introducing boundary condition"""

Th = 4.
Tc = 1.
T0 = 1.

