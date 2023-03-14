# -*- coding: utf-8 -*-
"""
FEM code for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import os
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
Time_tot = 6.3                  # Simulation length
sigma = 1

steady_state = False
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

def eval_source(x, y, t=0):
    """Heat source term [J.m^(-3).s^(-1)]"""
    assert x.shape == y.shape
    src = np.sin(x)*np.cos(2*y)*np.cos(t) - \
        (np.cos(x) - 5*(1 + x)*np.sin(x))*np.cos(2*y)*np.sin(t)
    return src

def eval_init_temperature(x, y):
    """Define initial temperature field"""
    # assert x.shape == y.shape
    # T_init = np.zeros(x.shape)
    # return T_init
    return eval_verify_field(x, y, 0)

def eval_verify_field(x, y, t):
    """Manufactured solution"""
    T_ref = np.sin(x)*np.cos(2*y)*np.sin(t)
    return T_ref

def dirichlet_bc(x, y, x_range, y_range, t=0, Tleft=0, Tright=0, Tbtm=0, Ttop=100):
    """Generate Dirichlet BC"""
    bc_idx_top = np.arange(y.size)[np.abs(y - y_range[1]) <= 1e-6]
    bc_idx_btm = np.arange(y.size)[np.abs(y - y_range[0]) <= 1e-6]
    bc_idx_left = np.arange(x.size)[np.abs(x - x_range[0]) <= 1e-6]
    bc_idx_right = np.arange(x.size)[np.abs(x - x_range[1]) <= 1e-6]
    bc_dof = np.concatenate([bc_idx_btm, bc_idx_top, bc_idx_left, bc_idx_right])
    # bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), Ttop*np.ones(bc_idx_top.shape)])
    # bc_dof = np.concatenate([bc_idx_btm, bc_idx_top, bc_idx_left, bc_idx_right])
    # bc_val = np.concatenate([Tbtm*np.ones(bc_idx_btm.shape), 
    #                          Ttop*np.ones(bc_idx_top.shape), 
    #                          Tleft*np.ones(bc_idx_left.shape), 
    #                          Tright*np.ones(bc_idx_right.shape)])
    T_ref = eval_verify_field(x, y, t)
    bc_val = np.concatenate([T_ref[bc_idx_btm], T_ref[bc_idx_top], T_ref[bc_idx_left], T_ref[bc_idx_right]])
    return bc_dof, bc_val


# Resolution
n_elem_trials = [4, 8, 16, 32, 64]
dt_trials = 2/np.array(n_elem_trials)
# n_elem_trials = [32, ]
L2_err_list = list()

# Quadrature configuration
xi_quad, wt_quad = quad.quad_2d["9-pt"]
n_quad = wt_quad.size

# Mesh-related config
element = mesh.BiQuadraticElement()

fig = plt.figure(figsize=(16, 6))

for n_elem_dim in n_elem_trials:
    
    dt = 0.1                    # Time stepping
    path_dir = "./output/diffuse_dynamic/biquad_{:.0e}_{:d}/".format(dt, n_elem_dim)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    """Mesh setup"""
    n_elem_x = n_elem_dim       # Number of elements in x dir
    n_elem_y = n_elem_dim
    n_elem_tot = n_elem_x*n_elem_y
    
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
    
    dx = Lx/n_elem_x            # Grid spacing
    dy = Ly/n_elem_y
    t_steps = np.arange(0, Time_tot, dt)
    
    """Initialization"""
    bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range)
    # Apply topography (optional)
    # np.random.seed(1105)
    # gcoord_nodes[1, bc_dof[-n_node_x:]] += 2*np.random.rand(n_node_x)
    # gcoord_nodes[1, bc_dof[n_node_x:2*n_node_x]] += 10*gcoord_x*(Lx - gcoord_x)/Lx**2

    T_init = eval_init_temperature(gcoord_nodes[0, :], gcoord_nodes[1, :])
    if bc_dof.size > 0:
        T_init[bc_dof] = bc_val
    # T_records = np.zeros((t_steps.size, T_init.size))
    # T_records[0, :] = T_init
    T_current = T_init
    L2_errors = np.zeros(t_steps.size)
    M_global = np.zeros((n_node_tot, n_node_tot))
    K_global = np.zeros((n_node_tot, n_node_tot))

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
        jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
        jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
        dN_vals_temp = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1)
        
        M_local = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
        M_local = np.sum(rho_temp*cp_temp*wt_quad*M_local*jac_det, axis=-1)
        
        K_local = np.stack([dN_vals_temp[:, :, i].T @ k_temp[i, :, :] @ dN_vals_temp[:, :, i] for i in range(n_quad)], axis=-1)
        K_local = np.sum(wt_quad*K_local*jac_det, axis=-1)
            
        idx = np.ix_(nodes_temp, nodes_temp)
        M_global[idx] += M_local
        K_global[idx] += K_local
    
    """Time-stepping"""
    for i_step, t_step in enumerate(t_steps):
        
        if i_step == 0:
            continue
        
        # time-dependent source
        F_global = np.zeros(n_node_tot)
        for i_elem in range(n_elem_tot):

            nodes_temp = connectivity[i_elem]
            coord_temp = gcoord_nodes[:, nodes_temp]
            pt_quad = coord_temp @ N_vals
            
            src_temp = eval_source(pt_quad[0, :], pt_quad[1, :], t=t_step)
            jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
            jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
            dN_vals_temp = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1)
            
            F_local = np.sum(wt_quad*N_vals*src_temp*jac_det, axis=-1)            
            F_global[nodes_temp] += F_local
        
        # Assemble system
        # T_prev = T_records[i_step-1, :]
        T_prev = T_current
        # Backward Euler
        L = M_global/dt + K_global
        b = M_global/dt @ T_prev + F_global
        # Forward Euler
        # L = M_global/dt
        # b = F_global + (M_global/dt - K_global) @ T_prev
        
        # Use a time-dependent BC
        bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range, t=t_step)

        # Apply BC (reduced matrix version)
        # L = sparse.csc_array(L)
        # b = b - L[:, bc_dof] @ np.asarray(bc_val)
        # rem_idx = [idx for idx in range(n_node_tot) if idx not in bc_dof]
        # b = b[rem_idx]
        # L = L[np.ix_(rem_idx, rem_idx)]

        # T = np.zeros(T_init.shape)
        # T[rem_idx] = splinalg.spsolve(L, b)
        # T[bc_dof] = np.asarray(bc_val)
        # T_records[i_step+1, :] = T
        
        b[bc_dof] = bc_val
        L[bc_dof, :] = 0
        L[np.ix_(bc_dof, bc_dof)] = np.eye(bc_dof.size, bc_dof.size)
        L = sparse.csc_array(L)
        T = splinalg.spsolve(L, b)
        T_current = T
        # T_records[i_step, :] = T
            
        # Unravel field
        # T_field = np.reshape(T_records[i_step, :], (n_node_y, n_node_x))
        T_field = np.reshape(T_current, (n_node_y, n_node_x))
        T_reference = eval_verify_field(gcoord_nodes[0, :], gcoord_nodes[1, :], t_step)
        
        # Integrated error
        L2_err = 0.0
        for i_elem in range(n_elem_tot):
            
            nodes_temp = connectivity[i_elem]
            coord_temp = gcoord_nodes[:, nodes_temp]
            
            pt_quad = coord_temp @ N_vals
            jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
            jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
            
            # T_est = T_records[i_step, nodes_temp] @ N_vals
            T_est = T_current[nodes_temp] @ N_vals
            T_ref = eval_verify_field(pt_quad[0], pt_quad[1], t_step)
            
            L2_err += np.sum((T_ref - T_est)**2*wt_quad*jac_det, axis=-1)

        L2_errors[i_step] = L2_err
                
        fig.clear()
        plt.subplot(1, 3, 1)
        plt.title("FEM")
        plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T_current, 
                        # 50, vmin=-1, vmax=+1,
                        np.linspace(-1, +1, 51), 
                        cmap="coolwarm", origin="lower", extend="both")
        plt.xlabel("x coordinate [m]", fontsize=14)
        plt.ylabel("y coordinate [m]", fontsize=14)
        plt.colorbar(ticks=np.linspace(-1, +1, 11), orientation="horizontal")
        plt.grid(which="both")
        plt.gca().axis("equal")
        
        plt.subplot(1, 3, 2)
        plt.title("Manufactured Solution")
        plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T_reference, 
                        # 50, vmin=-1, vmax=+1,
                        np.linspace(-1, +1, 51), 
                        cmap="coolwarm", origin="lower", extend="both")
        plt.xlabel("x coordinate [m]", fontsize=14)
        plt.ylabel("y coordinate [m]", fontsize=14)
        plt.colorbar(ticks=np.linspace(-1, +1, 11), orientation="horizontal")
        plt.grid(which="both")
        plt.gca().axis("equal")

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        T_error = T_reference - T_current
        plt.tricontourf(gcoord_nodes[0, :], gcoord_nodes[1, :], T_error, 
                        50, vmin=-np.max([0.1, np.abs(T_error).max()]), vmax=+np.max([0.1, np.abs(T_error).max()]),
                        cmap="coolwarm", origin="lower")
        plt.xlabel("x coordinate [m]", fontsize=14)
        plt.ylabel("y coordinate [m]", fontsize=14)
        plt.colorbar(orientation="horizontal")
        plt.grid(which="both")
        plt.gca().axis("equal")
        
        plt.suptitle("T = {:.2f}".format(t_steps[i_step]))
        plt.pause(0.1)
        
        if output:
            plt.savefig(os.path.join(path_dir, "snap_{:03d}.png".format(i_step)), format="png", bbox_inches="tight", dpi=150)

    # plt.show()

    # fig = plt.figure(figsize=(9, 6))
    # plt.plot(t_steps, L2_errors)
    # plt.xlabel("Time")
    # plt.ylabel("L2-error")
    # plt.grid(which="both")
    # plt.show()

    print("Elements = {:3d}, Integrated Error = {:.3e}".format(n_elem_dim, np.mean(L2_errors)))
    L2_err_list.append(np.mean(L2_errors))

if output:
    np.savez("./output/Diffuse_dyn_biquad_{:.0e}.npz".format(dt), n_elem_dim=np.asarray(n_elem_trials), l2_err=np.asarray(L2_err_list))
