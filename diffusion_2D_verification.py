# -*- coding: utf-8 -*-
"""
FEM code verification for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem import mesh, quad


"""Physics setup"""
x_range = [0, 10]               # Domain range
y_range = [0, 10]
Lx = x_range[1] - x_range[0]    # Domain width
Ly = y_range[1] - y_range[0]    # Domain height

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
    return np.multiply.outer(1 + x, np.eye(2))
    # return np.multiply.outer(np.ones(x.shape), np.diag([2, 0.5]))

def eval_source(x, y):
    """Heat source term [J.m^(-3).s^(-1)]"""
    assert x.shape == y.shape
    # return 1*np.ones(x.shape)
    return -np.cos(2*y)*(np.cos(x) - 5*(1 + x)*np.sin(x))

def eval_verify_field(x, y):
    return np.sin(x)*np.cos(2*y)

def eval_init_temperature(x, y):
    """Define initial temperature field"""
    assert x.shape == y.shape
    # T_init = Tmax*np.exp(-x**2/sigma**2)
    T_init = np.zeros(x.shape)
    return T_init

def dirichlet_bc(x, y, x_range, y_range, Tleft=0, Tright=0, Tbtm=0, Ttop=100):
    """Generate Dirichlet BC"""
    bc_idx_top = np.arange(y.size)[np.abs(y - y_range[1]) <= 1e-6]
    bc_idx_btm = np.arange(y.size)[np.abs(y - y_range[0]) <= 1e-6]
    bc_idx_left = np.arange(x.size)[np.abs(x - x_range[0]) <= 1e-6]
    bc_idx_right = np.arange(x.size)[np.abs(x - x_range[1]) <= 1e-6]
    
    # bc_dof = np.concatenate([bc_idx_btm, bc_idx_top])
    # bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), Ttop*np.ones(bc_idx_top.shape)])
    bc_dof = np.concatenate([bc_idx_btm, bc_idx_top, bc_idx_left, bc_idx_right])
    # bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), 
    #                          Ttop*np.ones(bc_idx_top.shape), 
    #                          Tleft*np.ones(bc_idx_left.shape), 
    #                          Tright*np.ones(bc_idx_right.shape)])
    T_ref = eval_verify_field(x, y)
    bc_val = np.concatenate([T_ref[bc_idx_btm], T_ref[bc_idx_top], 
                             T_ref[bc_idx_left], T_ref[bc_idx_right]])
    return bc_dof, bc_val


"""Mesh setup"""
# Quadrature configuration
xi_quad, wt_quad = quad.quad_2d["4-pt"]
n_quad = wt_quad.size

# Mesh-related config
order_mesh = 2
element = mesh.BiLinearElement()
N_vals = element.f_shape(xi_quad)
dN_vals = element.f_dshape(xi_quad)

# Resolution
n_elem_trials = [4, 8, 16, 32, 64]
L2_err_list = list()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 4.5))

for n_elem_dim in n_elem_trials:
    
    n_elem_x = n_elem_dim       # Number of elements in x dir
    n_elem_y = n_elem_dim
    n_elem_tot = n_elem_y*n_elem_x
    
    if isinstance(element, mesh.BiLinearElement):
        
        n_node_x = n_elem_x + 1     # Number of nodes in x dir
        n_node_y = n_elem_y + 1
        n_node_tot = n_node_y*n_node_x
        dx = Lx/n_elem_x            # Grid spacing
        dy = Ly/n_elem_y
        n_per_el = 4                # Nodes / element

        # Grid coordinates and connectivity
        gcoord_x = np.linspace(*x_range, num=n_node_x)
        gcoord_y = np.linspace(*y_range, num=n_node_y)
        gcoord_X, gcoord_Y = np.meshgrid(gcoord_x, gcoord_y, indexing='xy')
        gcoord_nodes = np.stack([gcoord_X.flatten(), gcoord_Y.flatten()], axis=0)
        connectivity = np.array([[n_node_x*i + j, n_node_x*(i+1) + j, n_node_x*(i+1) + (j+1), n_node_x*i + (j+1)] 
                                for i in range(n_elem_y) for j in range(n_elem_x)])
    
    elif isinstance(element, mesh.BiQuadraticElement):
        
        n_node_x = 2*n_elem_x + 1
        n_node_y = 2*n_elem_y + 1
        n_node_tot = n_node_x*n_node_y
        
        gcoord_x = np.linspace(*x_range, num=n_node_x)
        gcoord_y = np.linspace(*y_range, num=n_node_y)
        gcoord_X, gcoord_Y = np.meshgrid(gcoord_x, gcoord_y, indexing='xy')
        gcoord_nodes = np.stack([gcoord_X.flatten(), gcoord_Y.flatten()], axis=0)
        connectivity = np.array([[n_node_x*2*i + 2*j, n_node_x*2*(i+1) + 2*j, n_node_x*2*(i+1) + 2*(j+1), n_node_x*2*i + 2*(j+1), 
                                  n_node_x*(2*i+1) + 2*j, n_node_x*2*(i+1) + 2*j + 1, n_node_x*(2*i+1) + 2*(j+1), n_node_x*2*i + 2*j + 1, 
                                  n_node_x*(2*i+1) + 2*j + 1]
                                 for i in range(n_elem_y) for j in range(n_elem_x)])
    
    
    """Initialization"""
    bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range, Ttop=0)

    T_init = eval_init_temperature(gcoord_nodes[0, :], gcoord_nodes[1, :])
    if bc_dof.size > 0:
        T_init[bc_dof] = bc_val
    M_global = np.zeros((n_node_tot, n_node_tot))
    K_global = np.zeros((n_node_tot, n_node_tot))
    F_global = np.zeros(n_node_tot)


    """Solver"""
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
        
        F_local = np.sum(N_vals*src_temp*jac_det*wt_quad, axis=-1)
        
        idx = np.ix_(nodes_temp, nodes_temp)
        M_global[idx] += M_local
        K_global[idx] += K_local
        F_global[nodes_temp] += F_local
    
    L = K_global
    b = F_global

    # Apply BC (reduced matrix version)
    # L = sparse.csc_array(L)
    # b = b - L[:, bc_dof] @ np.asarray(bc_val)
    # rem_idx = [idx for idx in range(n_node_tot) if idx not in bc_dof]
    # b = b[rem_idx]
    # L = L[np.ix_(rem_idx, rem_idx)]

    # T = np.zeros(T_init.shape)
    # T[rem_idx] = splinalg.spsolve(L, b)
    # T[bc_dof] = np.asarray(bc_val)
    
    b[bc_dof] = bc_val
    L[bc_dof, :] = 0
    L[np.ix_(bc_dof, bc_dof)] = np.eye(bc_dof.size, bc_dof.size)
    L = sparse.csc_array(L)
    T = splinalg.spsolve(L, b)
    
    # Integrated error
    L2_err = 0.0
    for i_elem in range(n_elem_tot):
        
        nodes_temp = connectivity[i_elem]
        coord_temp = gcoord_nodes[:, nodes_temp]
        
        pt_quad = coord_temp @ N_vals
        jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
        jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
        
        T_ref = eval_verify_field(pt_quad[0, :], pt_quad[1, :])
        T_est = T[nodes_temp] @ N_vals
        
        L2_err += np.sum((T_ref - T_est)**2*wt_quad*jac_det, axis=-1)

    L2_err_list.append(L2_err)

    T_ref_nodes = eval_verify_field(gcoord_nodes[0, :], gcoord_nodes[1, :])
    clevels = np.linspace(np.min([T.min(), T_ref_nodes.min()]), np.max([T.max(), T_ref_nodes.max()]), 50)
    
    fig.clf()
    
    plt.subplot(1, 3, 1)
    plt.title("FEM", fontsize=18)
    plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T, 
                    clevels, 
                    cmap="coolwarm", origin="lower")
    plt.xlabel("x coordinate [m]", fontsize=16)
    plt.ylabel("y coordinate [m]", fontsize=16)
    plt.grid(which="both")
    plt.gca().axis("equal")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Manufactured Solution", fontsize=18)
    plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T_ref_nodes, 
                    # np.linspace(T_ref_nodes.min(), T_ref_nodes.max(), 50), 
                    clevels,
                    cmap="coolwarm", origin="lower")
    plt.xlabel("x coordinate [m]", fontsize=16)
    # plt.ylabel("y coordinate [m]", fontsize=14)
    plt.grid(which="both")
    plt.gca().axis("equal")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Difference", fontsize=18)
    diff_max = np.abs(T - T_ref_nodes).max()
    diff_levels = np.linspace(-diff_max, diff_max, num=50)
    plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T - T_ref_nodes, 
                    diff_levels,
                    cmap="coolwarm", origin="lower")
    plt.xlabel("x coordinate [m]", fontsize=16)
    # plt.ylabel("y coordinate [m]", fontsize=14)
    plt.grid(which="both")
    plt.gca().axis("equal")
    plt.colorbar()
    
    # plt.suptitle("{:d}*{:d} elements, element size {:f}".format(n_elem_dim, n_elem_dim, gcoord_x[1] - gcoord_x[0]), fontsize=22)
    if output:
        plt.savefig("./output/diffuse_static/BiLinear_{:}_elements.png".format(n_elem_dim), format="png", bbox_inches="tight", dpi=150)
    plt.pause(0.5)
    
for i in range(len(n_elem_trials)):
    print("{} - {:.3e}".format(n_elem_trials[i], L2_err_list[i]))

plt.show()
if output:
    np.savez("./L2_array_bilinear_sparse.npz", n_elem_dim=np.asarray(n_elem_trials), l2_err=np.asarray(L2_err_list))
