"""
Physical configuration for Torsional Oscillation
"""

import numpy as np


# Magnetic Prandtl number
Pm = 0.
# Lundquist number
Lu = 2000.
# Radial magnetic field profile
def Bs2_S1(s):
    return 63/184/np.pi*s**2*(1 - s**2)

def Bs2_S2(s):
    return 3/(28*182*16*np.pi)*s**2*(191222 - 734738*s**2 + 1060347*s**4 - 680108*s**6 + 163592*s**8)


"""Introducing coefficients"""

def m_u(s):
    return s**2*(1 - s**2)

def k_uu2(s):
    return Pm/Lu*s**2*(1 - s**2)

def k_uu1(s):
    return 3*Pm/Lu*s*(1 - s**2)

def k_ub1(s):
    return s*(1 - s**2)
    
def k_ub0(s):
    return 2 - 3*s**2

def m_b(s):
    return s

def k_bu1(s):
    return s**2*Bs2_S1(s)

def k_bb2(s):
    return s/Lu

def k_bb1(s):
    return -np.ones(s.shape)/Lu

def k_bb0(s):
    return np.zeros(s.shape)

# Necessary derivatives

def dk_uu2(s):
    return Pm/Lu*2*s*(1 - 2*s**2)

def dk_bb2(s):
    return np.ones(s.shape)/Lu
