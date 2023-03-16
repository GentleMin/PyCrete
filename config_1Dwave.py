"""Test case: standard wave equation
"""

import numpy as np


def m_u(s):
    return np.ones(s.shape)

def k_uu2(s):
    return np.zeros(s.shape)

def k_uu1(s):
    return np.zeros(s.shape)

def k_ub1(s):
    return np.ones(s.shape)
    
def k_ub0(s):
    return np.zeros(s.shape)

def k_bu1(s, Bs2):
    return np.ones(s.shape)

def m_b(s):
    return np.ones(s.shape)

def k_bb2(s):
    return np.zeros(s.shape)

def k_bb1(s):
    return np.zeros(s.shape)

def k_bb0(s):
    return np.zeros(s.shape)

def dk_uu2(s):
    return np.zeros(s.shape)

def dk_bb2(s):
    return np.zeros(s.shape)

