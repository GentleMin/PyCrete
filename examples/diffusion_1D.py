# -*- coding: utf-8 -*-
"""
FEM code for 1D diffusion equation, linear mesh (shape function)
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import matplotlib.pyplot as plt
import numpy as np


"""Physics setup"""
x_range = [-5, 5]               # Domain range
Lx = x_range[1] - x_range[0]    # Domain length
Time_tot = 60                   # Simulation length
Tmax = 100
sigma = 1

# Define thermal diffusivity structure [m2/s]
def get_kappa(x):
    return 1*np.ones(x.shape)

# Define heat source term [K/s]
def get_source(x):
    return 0*np.ones(x.shape)

# Define initial temperature profile
def get_temperature(x):
    return Tmax*np.exp(-x**2/sigma**2)

# Calculate analytical profile for a source-free infinite domain
# diffused from a Gaussian initial distribution
def calc_infty_model(x, t):
    return Tmax/np.sqrt(1 + 4*t*1/1)*np.exp(-x**2/(sigma**2 + 4*t*1))

# Calculate analytical profile for a steady-state solution
# with homogeneous source distribution
def calc_steady_state(x, t, src, kappa, T0, T1, Lx):
    return -1/2*src/kappa*(x + 5)**2 + (1/2*src*Lx/kappa + (T1 - T0)/Lx)*(x + 5) + T0

# Generate time-dependent BC
def generate_bc(g_coord, t):
    bc_dirichlet = True
    bc_dof = [0, len(g_coord) - 1]
    # bc_val = [0, 0]
    bc_val = calc_infty_model(np.asarray(g_coord[bc_dof]), t)
    return bc_dirichlet, bc_dof, bc_val

"""Mesh setup"""
n_elem = 9                  # Number of elements
n_node = n_elem + 1         # Number of nodes
n_per_el = 2                # Nodes / element
dx = Lx/n_elem              # Grid spacing
dt = 0.2                      # Time stepping
g_coord = np.linspace(x_range[0], x_range[1], num=n_node)       # Coordinate of grids
x_analytical = np.linspace(x_range[0], x_range[1], num=100)     # Coordinates for analytical solution
ele_coord = (g_coord[1:] + g_coord[:-1])/2      # Midpoint coordinate of elements
t_steps = np.arange(0, Time_tot, dt)            # Time steps
dx_elem = dx*np.ones(n_elem)                    # Element-wise dimension
kappa_elem = get_kappa(ele_coord)               # Element-wise diffusivity
src_elem = get_source(ele_coord)                # Element-wise source
# Connectivity matrix: indicating which nodes belong to each element
connectivity = np.array([[i, i+1] for i in range(n_elem)], dtype=int)

"""Boundary conditions setup"""
# Dirichlet
# bc_dirichlet = True
# bc_dof = [0, n_node - 1]
# bc_val = [0, 0]

"""Initial condition setup"""
T_init = get_temperature(g_coord)
bc_dirichlet, bc_dof, bc_val = generate_bc(g_coord, 0)
if bc_dirichlet:
    T_init[bc_dof] = bc_val
T_records = np.zeros((t_steps.size, T_init.size))
T_records[0, :] = T_init

"""Initializations"""
M_global = np.zeros((n_node, n_node))
K_global = np.zeros((n_node, n_node))
F_global = np.zeros(n_node)
M_local_base = np.array([[2, 1], [1, 2]])/6
K_local_base = np.array([[1, -1], [-1, 1]])
F_local_base = np.array([1, 1])

"""Assembling global matrices"""
for i_elem in range(n_elem):
    M_local = dx_elem[i_elem]*M_local_base
    K_local = (kappa_elem[i_elem]/dx_elem[i_elem])*K_local_base
    F_local = (src_elem[i_elem]*dx_elem[i_elem]/2)*F_local_base
    idx = np.ix_(connectivity[i_elem], connectivity[i_elem])
    M_global[idx] += M_local
    K_global[idx] += K_local
    F_global[connectivity[i_elem]] += F_local

# print(M_local)
# print(K_local)
# print(F_local)

fig = plt.figure(figsize=(9, 6))

"""Time-stepping"""
for i_step, t_step in enumerate(t_steps[:-1]):
    T_prev = T_records[i_step, :]
    L = M_global/dt + K_global
    b = M_global/dt @ T_prev + F_global
    
    # Apply BC (reduced matrix version)
    bc_dirichlet, bc_dof, bc_val = generate_bc(g_coord, t_steps[i_step+1])
    if bc_dirichlet:
        b -= L[:, bc_dof] @ np.asarray(bc_val)
        rem_idx = [idx for idx in range(n_node) if idx not in bc_dof]
        b = b[rem_idx]
        L = L[np.ix_(rem_idx, rem_idx)]
        
        # Solving
        T_rem = np.linalg.solve(L, b)
        T_records[i_step+1, rem_idx] = T_rem
        T_records[i_step+1, bc_dof] = np.asarray(bc_val)
    else:
        # Solving
        T_records[i_step+1, :] = np.linalg.solve(L, b)
    
    fig.clear()
    plt.title("T = {:.2f}".format(t_steps[i_step+1]))
    T_analytical = calc_infty_model(x_analytical, t_steps[i_step+1])
    # T_analytical = calc_steady_state(x_analytical, 0, 1, 1, 0, 0, Lx)
    plt.plot(x_analytical, T_analytical, 'k-', label="Analytical")    
    plt.plot(g_coord, T_records[i_step+1, :], 'ro', label="FEM")
    plt.xlabel("x coordinate [m]", fontsize=14)
    plt.xlim(x_range)
    plt.ylabel("Temperature [$^\circ$]", fontsize=14)
    # plt.ylim([0, max(14, Tmax)])
    plt.grid(which="both")
    plt.legend()
    plt.pause(0.1)

# print(T_records)
plt.show()

