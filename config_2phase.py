"""
Physical configuration for two-phase heat equation
"""

import numpy as np

diffuse_f = 0.05
diffuse_s = 0.08
couple_f = 5.
couple_s = 6.
background_flow = 1.


"""Introducing coefficients"""

def m_u(s):
    return np.ones(s.shape)

def k_uu2(s):
    return diffuse_f*np.ones(s.shape)

def k_uu1(s):
    return -background_flow*np.ones(s.shape)

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
T0 = 1.

def bound_left(t):
    return T0 + (Th - T0)/(1 + np.exp(-200*(t - 0.25)))
