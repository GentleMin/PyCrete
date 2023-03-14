# -*- coding: utf-8 -*-
"""
FEM code for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem_utils import mesh, quad


"""Physics setup"""
x_range = [0, 10]               # Domain range
y_range = [0, 10]
Lx = x_range[1] - x_range[0]    # Domain width
Ly = y_range[1] - y_range[0]    # Domain height
Time_tot = 50                   # Simulation length
sigma = 1

steady_state = True
output = False

def eval_density(x, y):
    """Medium density"""
    assert x.shape == y.shape
    return 1*np.ones(x.shape)

def eval_capacity(x, y):
    """Medium heat capacity [J.kg^(-1).K^(-1)]"""
    assert x.shape == y.shape
    return 1*np.ones(x.shape)

def eval_conductivity(x, y):
    """Medium thermal conductivity"""
    assert x.shape == y.shape
    return np.multiply.outer(np.ones(x.shape), np.eye(2))
    # return np.multiply.outer(np.ones(x.shape), np.diag([2, 0.5]))

def eval_source(x, y):
    """Heat source term [J.m^(-3).s^(-1)]"""
    assert x.shape == y.shape
    return 0*np.ones(x.shape)

def eval_init_temperature(x, y):
    """Define initial temperature field"""
    assert x.shape == y.shape
    T_init = np.zeros(x.shape)
    return T_init

def dirichlet_bc(x, y, x_range, y_range, Tleft=0, Tright=0, Tbtm=0, Ttop=100):
    """Generate Dirichlet BC"""
    bc_idx_top = np.arange(y.size)[np.abs(y - y_range[1]) <= 1e-6]
    bc_idx_btm = np.arange(y.size)[np.abs(y - y_range[0]) <= 1e-6]
    bc_idx_left = np.arange(x.size)[np.abs(x - x_range[0]) <= 1e-6]
    bc_idx_right = np.arange(x.size)[np.abs(x - x_range[1]) <= 1e-6]
    bc_dof = np.concatenate([bc_idx_btm, bc_idx_top])
    bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), Ttop*np.ones(bc_idx_top.shape)])
    # bc_dof = np.concatenate([bc_idx_btm, bc_idx_top, bc_idx_left, bc_idx_right])
    # bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), 
    #                          Ttop*np.ones(bc_idx_top.shape), 
    #                          Tleft*np.ones(bc_idx_left.shape), 
    #                          Tright*np.ones(bc_idx_right.shape)])
    return bc_dof, bc_val


"""Mesh setup"""
n_elem_x = 8                # Number of elements in x dir
n_elem_y = 8
n_elem_tot = n_elem_y*n_elem_x
n_node_x = n_elem_x + 1     # Number of nodes in x dir
n_node_y = n_elem_y + 1
n_node_tot = n_node_y*n_node_x
dx = Lx/n_elem_x            # Grid spacing
dy = Ly/n_elem_y
dt = 1                      # Time stepping
t_steps = np.arange(0, Time_tot, dt)
n_per_el = 4                # Nodes / element

# Grid coordinates and connectivity
gcoord_x = np.linspace(*x_range, num=n_node_x)
gcoord_y = np.linspace(*y_range, num=n_node_y)
gcoord_X, gcoord_Y = np.meshgrid(gcoord_x, gcoord_y, indexing='xy')
gcoord_nodes = np.stack([gcoord_X.flatten(), gcoord_Y.flatten()], axis=0)
connectivity = np.array([[n_node_x*i + j, n_node_x*(i+1) + j, n_node_x*(i+1) + (j+1), n_node_x*i + (j+1)] 
                         for i in range(n_elem_y) for j in range(n_elem_x)])

# print(connectivity)
# print(gcoord_nodes)

# Quadrature configuration
xi_quad, wt_quad = quad.quad_2d["4-pt"]
n_quad = wt_quad.size

# Mesh-related config
order_mesh = 2
element = mesh.BiLinearElement()


"""Initialization"""
bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range)
np.random.seed(1105)
# gcoord_nodes[1, bc_dof[-n_node_x:]] += 2*np.random.rand(n_node_x)
gcoord_nodes[1, bc_dof[n_node_x:2*n_node_x]] += 10*gcoord_x*(Lx - gcoord_x)/Lx**2

T_init = eval_init_temperature(gcoord_nodes[0, :], gcoord_nodes[1, :])
if bc_dof.size > 0:
    T_init[bc_dof] = bc_val
T_records = np.zeros((t_steps.size, T_init.size))
T_records[0, :] = T_init
M_global = np.zeros((n_node_tot, n_node_tot))
K_global = np.zeros((n_node_tot, n_node_tot))
F_global = np.zeros(n_node_tot)

N_vals = element.f_shape(xi_quad)
dN_vals = element.f_dshape(xi_quad)


"""Assembling global matrices"""
for i_elem in range(n_elem_tot):
    
    nodes_temp = connectivity[i_elem]
    coord_temp = gcoord_nodes[:, nodes_temp]
    
    pt_quad = coord_temp @ N_vals
    
    rho_temp = eval_density(pt_quad[0, :], pt_quad[1, :])
    cp_temp = eval_capacity(pt_quad[0, :], pt_quad[1, :])
    k_temp = eval_conductivity(pt_quad[0, :], pt_quad[1, :])
    src_temp = eval_source(pt_quad[0, :], pt_quad[1, :])
    jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
    jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
    dN_vals_temp = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1)
    
    M_local = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    M_local = np.sum(rho_temp*cp_temp*wt_quad*M_local*jac_det, axis=-1)
    
    K_local = np.stack([dN_vals_temp[:, :, i].T @ k_temp[i, :, :] @ dN_vals_temp[:, :, i] for i in range(n_quad)], axis=-1)
    K_local = np.sum(wt_quad*K_local*jac_det, axis=-1)
    
    F_local = np.sum(wt_quad*N_vals*src_temp*jac_det, axis=-1)
    
    idx = np.ix_(nodes_temp, nodes_temp)
    M_global[idx] += M_local
    K_global[idx] += K_local
    F_global[nodes_temp] += F_local


if steady_state:
    
    L = K_global
    b = F_global
    
    # Apply BC (reduced matrix version)
    L = sparse.csc_array(L)
    b = b - L[:, bc_dof] @ np.asarray(bc_val)
    rem_idx = [idx for idx in range(n_node_tot) if idx not in bc_dof]
    b = b[rem_idx]
    L = L[np.ix_(rem_idx, rem_idx)]

    # Solve
    T = np.zeros(T_init.shape)
    T[rem_idx] = splinalg.spsolve(L, b)
    T[bc_dof] = np.asarray(bc_val)
    
    # b[bc_dof] = bc_val
    # L[bc_dof, :] = 0
    # L[np.ix_(bc_dof, bc_dof)] = np.eye(bc_dof.size, bc_dof.size)
    # L = sparse.csc_array(L)
    # T = splinalg.spsolve(L, b)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    
    if not output:
        plt.title("Steady-state solution", fontsize=18)
    plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T, 
                    np.linspace(T.min(), T.max(), 50), 
                    cmap="coolwarm", origin="lower")
    plt.xlabel("x coordinate [m]", fontsize=14)
    plt.ylabel("y coordinate [m]", fontsize=14)
    plt.colorbar(ticks=np.linspace(0, 100, 11))
    plt.grid(which="both")
    plt.gca().axis("equal")
    if output:
        plt.savefig("./output/diffuse_static/topo_solution_{:d}{:d}.png".format(n_elem_x, n_elem_y), format="png", bbox_inches="tight", dpi=150)
    plt.show()

else:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    """Time-stepping"""
    for i_step, t_step in enumerate(t_steps[:-1]):
        T_prev = T_records[i_step, :]
        L = M_global/dt + K_global
        b = M_global/dt @ T_prev + F_global
        
        # Use a time-dependent BC
        # bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range)
        
        # Apply BC (reduced matrix version)
        b -= L[:, bc_dof] @ np.asarray(bc_val)
        rem_idx = [idx for idx in range(n_node_tot) if idx not in bc_dof]
        b = b[rem_idx]
        L = L[np.ix_(rem_idx, rem_idx)]
        
        # Solving
        T_rem = np.linalg.solve(L, b)
        T_records[i_step+1, rem_idx] = T_rem
        T_records[i_step+1, bc_dof] = np.asarray(bc_val)
        
        # Unravel field
        T_field = np.reshape(T_records[i_step+1, :], (n_node_y, n_node_x))
        
        fig.clear()
        plt.title("T = {:.2f}".format(t_steps[i_step+1]), fontsize=18)
        plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T_records[i_step+1, :], 
                        np.linspace(T_records[i_step+1, :].min(), T_records[i_step+1, :].max(), 50), 
                        cmap="coolwarm", origin="lower")
        plt.xlabel("x coordinate [m]", fontsize=14)
        plt.ylabel("y coordinate [m]", fontsize=14)
        plt.colorbar(ticks=np.linspace(0, 100, 11))
        plt.grid(which="both")
        plt.gca().axis("equal")
        if output:
            plt.savefig("./output/diffuse_dynamic/topo_snap_{:d}{:d}_{:02d}.png".format(n_elem_x, n_elem_y, i_step), 
                        format="png", bbox_inches="tight", dpi=150)
        plt.pause(0.1)

    plt.show()

