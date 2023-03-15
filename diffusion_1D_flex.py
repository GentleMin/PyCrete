# -*- coding: utf-8 -*-
"""
FEM code for 1D diffusion equation, customized shape functions and numerical quadrature
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import matplotlib.pyplot as plt
import numpy as np


"""Physics setup"""
x_range = [-5, 5]               # Domain range
Lx = x_range[1] - x_range[0]    # Domain length
Time_tot = 2                  # Simulation length
Tmax = 0
sigma = 1

# Define thermal diffusivity structure [m2/s]
def eval_kappa(x):
    return 1*np.ones(x.shape)

# Define heat source term [K/s]
def eval_source(x):
    return 1*np.ones(x.shape)

# Define initial temperature profile
def eval_init_temperature(x):
    return Tmax*np.exp(-x**2/sigma**2)

# Calculate analytical profile for a source-free infinite domain
# diffused from a Gaussian initial distribution
def calc_infty_model(x, t):
    return Tmax/np.sqrt(1 + 4*t*1/1)*np.exp(-x**2/(sigma**2 + 4*t*1))

# Calculate analytical profile for a steady-state solution
# with homogeneous source distribution
def calc_steady_state(x, src, kappa, T0, T1, Lx):
    return -1/2*src/kappa*(x + 5)**2 + (1/2*src*Lx/kappa + (T1 - T0)/Lx)*(x + 5) + T0

# Generate time-dependent BC
def generate_bc(g_coord, t, g_coord_tot=None):
    bc_dirichlet = True
    if g_coord_tot is not None:
        bc_dof = [0, g_coord_tot.size - 1]
    else:
        bc_dof = [0, g_coord.size - 1]
    bc_val = [0, 0]
    return bc_dirichlet, bc_dof, bc_val


"""Mesh setup"""
n_elem = 9                  # Number of elements
n_node = n_elem + 1         # Number of nodes
n_per_el = 2                # Nodes / element
dx = Lx/n_elem              # Grid spacing
dt = 1                      # Time stepping

# Quadrature-related config
n_quad = 1
order_quad = 2*n_quad - 1
xi_quad = np.array([-1, 1])/np.sqrt(3)
wt_quad = np.array([1, 1])
# xi_quad = np.array([-0.774596669241483, 0, +0.774596669241483])
# wt_quad = np.array([5, 8, 5])/9

# Mesh-related config
order_mesh = 2

def f_shape(xi):
    N_1 = (1 - xi)/2
    N_2 = (1 + xi)/2
    return N_1, N_2

def f_dshape(xi):
    dN_1 = -1/2
    dN_2 = 1/2
    return dN_1, dN_2

def f_shape(xi):
    N_1 = 1/2*xi*(xi - 1)
    N_2 = 1 - xi*xi
    N_3 = 1/2*xi*(xi + 1)
    return N_1, N_2, N_3

def f_dshape(xi):
    dN_1 = xi - 1/2
    dN_2 = -2*xi
    dN_3 = xi + 1/2
    return dN_1, dN_2, dN_3

def coord_mapping(xi, g_coord, i_elem):
    return g_coord[i_elem] + (1 + xi)/2*(g_coord[i_elem+1] - g_coord[i_elem])

g_coord = np.linspace(x_range[0], x_range[1], num=n_node)       # Coordinate of grids
x_analytical = np.linspace(x_range[0], x_range[1], num=100)     # Coordinates for analytical solution
t_steps = np.arange(0, Time_tot, dt)            # Time steps
dx_elem = dx*np.ones(n_elem)                    # Element-wise dimension
jacobian = dx_elem/2                            # Element-wise Jacobian

n_node_tot = n_elem*(order_mesh + 1) - (n_elem - 1)
g_coord_tot = np.linspace(x_range[0], x_range[1], num=n_node_tot)

# Connectivity matrix: indicating which nodes belong to each element
# connectivity = np.array([[i, i+1] for i in range(n_elem)], dtype=int)
connectivity = np.array([[2*i, 2*i+1, 2*i+2] for i in range(n_elem)], dtype=int)

"""Initial condition setup"""
T_init = eval_init_temperature(g_coord_tot)
bc_dirichlet, bc_dof, bc_val = generate_bc(g_coord, 0, g_coord_tot=g_coord_tot)
if bc_dirichlet:
    T_init[bc_dof] = bc_val
T_records = np.zeros((t_steps.size, T_init.size))
T_records[0, :] = T_init

"""Initializations"""
M_global = np.zeros((n_node_tot, n_node_tot))
K_global = np.zeros((n_node_tot, n_node_tot))
F_global = np.zeros(n_node_tot)

"""Assembling global matrices"""
for i_elem in range(n_elem):
    
    N_vals = f_shape(xi_quad)
    dN_vals = f_dshape(xi_quad)
    kappa_val = eval_kappa(coord_mapping(xi_quad, g_coord, i_elem))
    src_val = eval_source(coord_mapping(xi_quad, g_coord, i_elem))
    
    M_local = np.array([[np.sum(wt_quad*N_vals[i]*N_vals[j]*jacobian[i_elem]) for j in range(order_mesh+1)] for i in range(order_mesh+1)])
    K_local = np.array([[np.sum(wt_quad*kappa_val*dN_vals[i]*dN_vals[j]/jacobian[i_elem]) for j in range(order_mesh+1)] for i in range(order_mesh+1)])
    F_local = np.array([np.sum(wt_quad*N_vals[i]*src_val*jacobian[i_elem]) for i in range(order_mesh+1)])
    
    idx = np.ix_(connectivity[i_elem], connectivity[i_elem])
    M_global[idx] += M_local
    K_global[idx] += K_local
    F_global[connectivity[i_elem]] += F_local

fig = plt.figure(figsize=(9, 6))

# print(dx_elem)
# print(jacobian)
# print(M_local)
# print(K_local)
# print(F_local)

"""Time-stepping"""
for i_step, t_step in enumerate(t_steps[:-1]):
    T_prev = T_records[i_step, :]
    # L = M_global/dt + K_global
    # b = M_global/dt @ T_prev + F_global
    L = K_global
    b = F_global
    
    bc_dirichlet, bc_dof, bc_val = generate_bc(g_coord, t_steps[i_step+1], g_coord_tot=g_coord_tot)
    # Apply BC (reduced matrix version)
    if bc_dirichlet:
        b -= L[:, bc_dof] @ np.asarray(bc_val)
        rem_idx = [idx for idx in range(n_node_tot) if idx not in bc_dof]
        b = b[rem_idx]
        L = L[np.ix_(rem_idx, rem_idx)]
        
        print(np.linalg.cond(L))
        
        # Solving
        T_rem = np.linalg.solve(L, b)
        T_records[i_step+1, rem_idx] = T_rem
        T_records[i_step+1, bc_dof] = np.asarray(bc_val)
    else:
        # Solving
        T_records[i_step+1, :] = np.linalg.solve(L, b)
    
    fig.clear()
    plt.title("T = {:.2f}".format(t_steps[i_step+1]))
    # T_analytical = calc_infty_model(x_analytical, t_steps[i_step+1])
    T_analytical = calc_steady_state(x_analytical, 1, 1, 0, 0, Lx)
    plt.plot(x_analytical, T_analytical, 'k-', label="Analytical")    
    plt.plot(g_coord_tot, T_records[i_step+1, :], 'ro', label="FEM")
    plt.xlabel("x coordinate [m]", fontsize=14)
    plt.xlim(x_range)
    plt.ylabel("Temperature [$^\circ$]", fontsize=14)
    plt.ylim([0, max(14, Tmax)])
    plt.grid(which="both")
    plt.legend()
    plt.pause(0.1)

# print(T_records)
plt.show()

T_compare = calc_steady_state(g_coord_tot, 1, 1, 0, 0, Lx)
# print(np.abs(T_compare - T_records[-1, :]))
