# -*- coding: utf-8 -*-
"""
FEM code for 2D Stokes equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem import mesh, quad


"""Configurations"""
verbose = True
output = False
dynamic = True


"""Physics setup"""

# Domain
x_range = [0, 10]
y_range = [0, 10]
t_range = 200.0
Lx = x_range[1] - x_range[0]
Ly = y_range[1] - y_range[0]

def iso_viscosity(x, y, mu=1e+6):
    """Isotropic viscosity matrix"""
    assert x.shape == y.shape
    D = mu*np.diag([2, 2, 1])
    return np.multiply.outer(np.ones(x.shape), D)

def init_interface(x, y, x_range, y_range, h=3, A0=1):
    """Define interface for Rayleigh-Taylor instability
    """
    assert x.shape == y.shape
    return y <= h - A0*np.sin(np.pi*(x - x_range[0])/(x_range[1] - x_range[0]))

def init_sphere(x, y, x_range, y_range, center=(5, 4), R0=2):
    """Define sphere for intrusion"""
    assert x.shape == y.shape
    return (x - center[0])**2 + (y - center[1])**2 <= R0**2

def calc_density(x, y, rho1=5e+3, rho2=2e+3):
    """Density
    """
    assert x.shape == y.shape
    # idx_part = init_interface(x, y, x_range, y_range, h=5, A0=2)
    idx_part = init_sphere(x, y, x_range, y_range)
    rho = rho1*np.ones(x.shape)
    rho[idx_part] = rho2
    return rho

def calc_traction(x, y):
    """Traction
    """
    assert x.shape == y.shape
    return np.multiply.outer(np.ones(x.shape), np.array([0, 0]))

def calc_body_force(x, y):
    """Body force
    """
    rho = calc_density(x, y)
    return np.multiply.outer(rho, np.array([0, -10]))

def get_rectangular_boundaries(x, y, x_range, y_range):
    """Find boundaries according to a rectangular domain
    """
    bc_idx_top = np.arange(y.size)[np.abs(y - y_range[1]) <= 1e-6]
    bc_idx_btm = np.arange(y.size)[np.abs(y - y_range[0]) <= 1e-6]
    bc_idx_left = np.arange(x.size)[np.abs(x - x_range[0]) <= 1e-6]
    bc_idx_right = np.arange(x.size)[np.abs(x - x_range[1]) <= 1e-6]
    return bc_idx_top, bc_idx_btm, bc_idx_left, bc_idx_right

def free_slip_bc(x, y, x_range, y_range, node2dof):
    """Generate free-slip boundary condition
    """
    bc_idx_top, bc_idx_btm, bc_idx_left, bc_idx_right = get_rectangular_boundaries(x, y, x_range, y_range)
    bc_dof = np.concatenate([node2dof[bc_idx_top, 1], node2dof[bc_idx_btm, 1], 
                             node2dof[bc_idx_left, 0], node2dof[bc_idx_right, 0]])
    bc_val = np.zeros(bc_dof.shape)
    return bc_dof, bc_val

def no_slip_bc(x, y, x_range, y_range, node2dof):
    """Generate no-slip boundary condition
    """
    _, _, bc_idx_left, bc_idx_right = get_rectangular_boundaries(x, y, x_range, y_range)
    bc_dof = np.concatenate([node2dof[bc_idx_left, 0], node2dof[bc_idx_right, 0], 
                             node2dof[bc_idx_left, 1], node2dof[bc_idx_right, 1]])
    bc_val = np.zeros(bc_dof.size)
    return bc_dof, bc_val

def pressure_bc(x, y, x_range, y_range):
    # bc_idx = np.arange(x.size)[(np.abs(x - x_range[0]) <= 1e-6) & (np.abs(y - y_range[1]) <= 1e-6)]
    # bc_dof = np.atleast_1d(bc_idx)
    # bc_val = np.zeros(bc_dof.shape)
    bc_idx_top, _, _, _ = get_rectangular_boundaries(x, y, x_range, y_range)
    bc_dof = bc_idx_top
    bc_val = np.zeros(bc_dof.size)
    return bc_dof, bc_val


"""Mesh setup"""

n_elem_x =  16               # Number of elements in x dir
n_elem_y =  16
n_elem_tot = n_elem_y*n_elem_x
dx = Lx/n_elem_x            # Element dimensions
dy = Ly/n_elem_y
dt_max = 10.0
step_max = 500
t_steps = [0,]

ndof_node_v = 2
ndof_node_p = 1

# Quadrature configuration
xi_quad, wt_quad = quad.quad_2d["9-pt"]
n_quad = wt_quad.size
print("Using 9-pt quadrature.")

# Velocity element
elem_v = mesh.BiQuadraticElement()
nodes_elem_v = elem_v.n_nodes
order_elem_v = elem_v.order
print("Using {} for velocity.".format(type(elem_v).__name__))

# Pressure element
elem_p = mesh.BiLinearElement()
nodes_elem_p = elem_p.n_nodes
order_elem_p = elem_p.order
print("Using {} for pressure.".format(type(elem_p).__name__))

# Complete mesh (Velocity mesh)
n_node_x_v = 2*n_elem_x + 1
n_node_y_v = 2*n_elem_y + 1
n_node_tot_v = n_node_x_v*n_node_y_v
n_dof_v = ndof_node_v*n_node_tot_v
# Coordinates
xcoord_v = np.linspace(*x_range, num=n_node_x_v)
ycoord_v = np.linspace(*y_range, num=n_node_y_v)
Xcoord_v, Ycoord_v = np.meshgrid(xcoord_v, ycoord_v, indexing='xy')
gcoord_v = np.stack([Xcoord_v.flatten(), Ycoord_v.flatten()], axis=0)
gcoord_v_original = gcoord_v.copy()
# Connections
elem2node_v = np.array([[n_node_x_v*2*i + 2*j, n_node_x_v*2*(i+1) + 2*j, n_node_x_v*2*(i+1) + 2*(j+1), n_node_x_v*2*i + 2*(j+1), 
                         n_node_x_v*(2*i+1) + 2*j, n_node_x_v*2*(i+1) + (2*j+1), n_node_x_v*(2*i+1) + 2*(j+1), n_node_x_v*2*i + (2*j+1), 
                         n_node_x_v*(2*i+1) + (2*j+1)] for i in range(n_elem_y) for j in range(n_elem_x)])
node2dof_v = np.stack([2*np.arange(n_node_tot_v), 2*np.arange(n_node_tot_v) + 1], axis=-1)
elem2dof_v = np.array([node2dof_v[elem_nodes, :].flatten() for elem_nodes in elem2node_v])

# Pressure mesh subset
n_node_x_p = n_elem_x + 1
n_node_y_p = n_elem_y + 1
n_node_tot_p = n_node_y_p*n_node_x_p
n_dof_p = ndof_node_p*n_node_tot_p
# Subset slice
xslice_p = range(0, n_node_x_v, 2)
yslice_p = range(0, n_node_y_v, 2)
node_idx_p = np.array([n_node_x_v*iy + ix for iy in yslice_p for ix in xslice_p])
gcoord_p = gcoord_v[:, node_idx_p]
gcoord_p_original = gcoord_p.copy()
# Connections
elem2node_p = np.array([[n_node_x_p*i + j, n_node_x_p*(i+1) + j, n_node_x_p*(i+1) + (j+1), n_node_x_p*i + (j+1)] 
                        for i in range(n_elem_y) for j in range(n_elem_x)])
elem2dof_p = elem2node_p

n_dof_tot = n_dof_v + n_dof_p


"""Initialization"""

# velocity element shape functions
N_vals = elem_v.f_shape(xi_quad)
dN_vals = elem_v.f_dshape(xi_quad)
N = np.concatenate([np.array([[N_vals[i], np.zeros(N_vals[i].shape)], 
                              [np.zeros(N_vals[i].shape), N_vals[i]]]) for i in range(nodes_elem_v)], axis=1)

# pressure element shape functions
Np = elem_p.f_shape(xi_quad)
dNp_vals = elem_p.f_dshape(xi_quad)
em = np.array([1, 1, 0])

# BC
bc_dof_v, bc_val_v = free_slip_bc(gcoord_v[0, :], gcoord_v[1, :], x_range, y_range, node2dof_v)
# bc_dof_v, bc_val_v = no_slip_bc(gcoord_v[0, :], gcoord_v[1, :], x_range, y_range, node2dof_v)
bc_dof_p, bc_val_p = pressure_bc(gcoord_p[0, :], gcoord_p[1, :], x_range, y_range)
bc_dof = np.r_[bc_dof_v, n_dof_v + bc_dof_p]
bc_val = np.r_[bc_val_v, bc_val_p]

"""Time-stepping"""


# Setup figure
fig = plt.figure(figsize=(12, 10))
fig_flow = plt.figure(figsize=(8, 6))

# for each time step
for i_time in range(step_max):
        
    Kvv = np.zeros((n_dof_v, n_dof_v))
    Kvp = np.zeros((n_dof_v, n_dof_p))
    F = np.zeros(n_dof_v)
    
    # Assembling global matrices
    # Note that for time-independent load and elasticity and small deformations, the global matrices need not be reformed every time step
    for i_elem in range(n_elem_tot):
        
        nodes_temp_v = elem2node_v[i_elem]
        coord_temp_v = gcoord_v[:, nodes_temp_v]
        coord_origin = gcoord_v_original[:, nodes_temp_v]
        dof_temp_v = elem2dof_v[i_elem]
        dof_temp_p = elem2dof_p[i_elem]
        
        pt_quad = coord_temp_v @ N_vals
        pt_quad_origin = coord_origin @ N_vals
        
        viscosity = iso_viscosity(pt_quad[0, :], pt_quad[1, :])
        body_force = calc_body_force(pt_quad_origin[0, :], pt_quad_origin[1, :])
        
        jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp_v.T for i in range(n_quad)], axis=-1)
        jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
        dN_dx = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1).transpose((1, 0, 2))
        B_mat = np.concatenate([np.array([[dN_dx[i, 0], np.zeros(n_quad)], 
                                        [np.zeros(n_quad), dN_dx[i, 1]],
                                        [dN_dx[i, 1], dN_dx[i, 0]]]) for i in range(nodes_elem_v)], axis=1)
        
        Kvv_local = np.stack([B_mat[:, :, i].T @ viscosity[i, :, :] @ B_mat[:, :, i] for i in range(n_quad)], axis=-1)
        Kvv_local = np.sum(wt_quad*Kvv_local*jac_det, axis=-1)
        
        Kvp_local = np.stack([B_mat[:, :, i].T @ np.outer(em, Np[:, i]) for i in range(n_quad)], axis=-1)
        Kvp_local = - np.sum(wt_quad*Kvp_local*jac_det, axis=-1)
        
        F_local = np.stack([N[:, :, i].T @ body_force[i] for i in range(n_quad)], axis=-1)
        F_local = np.sum(wt_quad*F_local*jac_det, axis=-1)
        
        idx_vv = np.ix_(dof_temp_v, dof_temp_v)
        idx_vp = np.ix_(dof_temp_v, dof_temp_p)
        Kvv[idx_vv] += Kvv_local
        Kvp[idx_vp] += Kvp_local
        F[dof_temp_v] += F_local
    
    K_global = np.block([[Kvv, Kvp], 
                         [Kvp.T, np.zeros((n_dof_p, n_dof_p))]])
    F_global = np.concatenate([F, np.zeros(n_dof_p)])
    
    L = K_global
    b = F_global
    if verbose:
        print("Time step {:3d}, cond(L) = {:.2e}".format(i_time, np.linalg.cond(L)))
    
    # Apply BC (reduced matrix version)
    # L = sparse.csc_array(L)
    # b -= L[:, bc_dof] @ np.asarray(bc_val)
    # rem_idx = np.array([idx for idx in range(n_dof_tot) if idx not in bc_dof])
    # b = b[rem_idx]
    # L = L[np.ix_(rem_idx, rem_idx)]

    # vp_solution = splinalg.spsolve(L, b)
    # v_global = np.zeros(n_dof_v)
    # v_global[rem_idx[:n_dof_v-bc_dof_v.size]] = vp_solution[:n_dof_v-bc_dof_v.size]
    # v_global[bc_dof_v] = bc_val_v
    # p = np.zeros(n_dof_p)
    # p[rem_idx[-n_dof_p+bc_dof_p.size:]-n_dof_v] = vp_solution[-n_dof_p+bc_dof_p.size:]
    # p[bc_dof_p] = bc_val_p
    
    # Apply BC (full matrix version)
    L[bc_dof, :] = 0
    L[np.ix_(bc_dof, bc_dof)] = np.eye(bc_dof.size, bc_dof.size)
    b[bc_dof] = bc_val
    rem_idx = [idx for idx in range(n_dof_tot) if idx not in bc_dof]
    L = sparse.csc_array(L)
    vp_solution = splinalg.spsolve(L, b)
    v_global = vp_solution[:n_dof_v]
    p = vp_solution[-n_dof_p:]
        
    """Unravel fields and visualize"""

    vx = v_global[node2dof_v[:, 0]]
    vy = v_global[node2dof_v[:, 1]]
    
    dt = np.min([dx, dy])/np.abs(v_global).max()/4
    dt = np.min([dt, dt_max])
    
    # Extra diagnostic fields
    quad_coord = np.zeros((2, n_elem_tot*n_quad))
    # Density
    quad_dense = np.zeros((n_elem_tot*n_quad))
    # Displacement and velocity
    quad_vel = np.zeros((2, n_elem_tot*n_quad))
    # Stress and strain
    quad_stress_dev = np.zeros((3, n_elem_tot*n_quad))
    quad_strain_rate = np.zeros((3, n_elem_tot*n_quad))
    
    for i_elem in range(n_elem_tot):
        
        nodes_temp_v = elem2node_v[i_elem]
        coord_temp_v = gcoord_v[:, nodes_temp_v]
        coord_origin = gcoord_v_original[:, nodes_temp_v]
        dof_temp_v = elem2dof_v[i_elem]
        
        pt_quad = coord_temp_v @ N_vals
        pt_quad_origin = coord_origin @ N_vals
        quad_coord[:, i_elem*n_quad:(i_elem+1)*n_quad] = pt_quad
        
        # Density
        quad_dense[i_elem*n_quad:(i_elem+1)*n_quad] = calc_density(pt_quad_origin[0, :], pt_quad_origin[1, :])
        
        # Velocity
        v_nodes = np.array([vx[nodes_temp_v], vy[nodes_temp_v]]).flatten(order='F')
        quad_vel[0, i_elem*n_quad:(i_elem+1)*n_quad] = vx[nodes_temp_v] @ N_vals
        quad_vel[1, i_elem*n_quad:(i_elem+1)*n_quad] = vy[nodes_temp_v] @ N_vals
        viscosity = iso_viscosity(pt_quad[0, :], pt_quad[1, :])
        
        jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp_v.T for i in range(n_quad)], axis=-1)
        jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
        dN_dx = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1).transpose((1, 0, 2))
        B_mat = np.concatenate([np.array([[dN_dx[i, 0], np.zeros(n_quad)], 
                                        [np.zeros(n_quad), dN_dx[i, 1]],
                                        [dN_dx[i, 1], dN_dx[i, 0]]]) for i in range(nodes_elem_v)], axis=1)
        
        strain_rate_temp = np.stack([B_mat[:, :, i] @ v_nodes for i in range(n_quad)], axis=-1)
        stress_dev_temp = np.stack([viscosity[i, :, :] @ strain_rate_temp[:, i] for i in range(n_quad)], axis=-1)
        quad_strain_rate[:, i_elem*n_quad:(i_elem+1)*n_quad] = strain_rate_temp
        quad_stress_dev[:, i_elem*n_quad:(i_elem+1)*n_quad] = stress_dev_temp
    
    # 2nd invariant of deviatoric stress
    dev_stress_invar2 = np.sqrt(quad_stress_dev[2, :]**2 + (quad_stress_dev[0, :] - quad_stress_dev[1, :])**2/4)
    # Kinetic / Elastic / Gravity potential energy densities
    quad_e_kinetic = np.linalg.norm(quad_vel, axis=0)**2/2
    # quad_e_elastic = np.sum(quad_strain*quad_stress, axis=0)/2
    # quad_e_gravity = quad_density*10*quad_disp[1, :]
    
    
    """Visualization"""
    fig.clear()
    axes = fig.subplots(nrows=3, ncols=3)
    
    # Velocity - x
    ax = axes[0, 0]
    contour = ax.tricontourf(gcoord_v[0, :], gcoord_v[1, :], vx, 
                            50, vmin=-np.abs(vx).max(), vmax=np.abs(vx).max(),
                            cmap="coolwarm", origin="lower")
    ax.set_ylabel("y coordinate [m]", fontsize=14)
    ax.set_title(r"$v_x$ [m]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Velocity - y
    ax = axes[0, 1]
    contour = ax.tricontourf(gcoord_v[0, :], gcoord_v[1, :], vy, 
                            50, vmin=-np.abs(vy).max(), vmax=np.abs(vy).max(),
                            cmap="coolwarm", origin="lower")
    ax.set_title(r"$v_y$ [m]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Hydrostatic pressure
    ax = axes[0, 2]
    contour = ax.tricontourf(gcoord_p[0, :], gcoord_p[1, :], p, 
                             50, 
                             cmap="plasma", origin="lower")
    ax.set_title("p [Pa]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Strain rate xx (tensile positive)
    ax = axes[1, 0]
    contour = ax.tricontourf(quad_coord[0, :], 
                             quad_coord[1, :], 
                             quad_strain_rate[0, :], 
                             50, 
                             vmin=-np.abs(quad_strain_rate[0, :]).max(), vmax = np.abs(quad_strain_rate[0, :]).max(),
                             cmap="Spectral", origin="lower")
    ax.set_ylabel("y coordinate [m]", fontsize=14)
    ax.set_title(r"$\dot{\epsilon}_{xx}$ [s$^{-1}$]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Strain rate yy (tensile positive)
    ax = axes[1, 1]
    contour = ax.tricontourf(quad_coord[0, :], 
                             quad_coord[1, :], 
                             quad_strain_rate[1, :], 
                             50, 
                             vmin=-np.abs(quad_strain_rate[1, :]).max(), vmax = np.abs(quad_strain_rate[1, :]).max(),
                             cmap="Spectral", origin="lower")
    ax.set_title(r"$\dot{\epsilon}_{yy}$ [s$^{-1}$]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Engineering shear stress gamma_xy = 2*sigma_xy
    ax = axes[1, 2]
    contour = ax.tricontourf(quad_coord[0, :], 
                             quad_coord[1, :], 
                             quad_strain_rate[2, :]/2, 
                            50,
                            cmap="Spectral", origin="lower")
    ax.set_title(r"$\dot{\varepsilon}_{xy}$ [s$^{-1}$]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # Kinetic energy density
    ax = axes[2, 0]
    contour = ax.tricontourf(quad_coord[0, :], quad_coord[1, :], 
                             quad_e_kinetic, 50,
                             cmap="plasma", origin="lower")
    ax.set_xlabel("x coordinate [m]", fontsize=14)
    ax.set_ylabel("y coordinate [m]", fontsize=14)
    ax.set_title(r"$E_k$ density [$J/m^3$]")
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    # 2nd invariant of deviatoric stress
    ax = axes[2, 1]
    contour = ax.tricontourf(quad_coord[0, :], 
                             quad_coord[1, :], 
                             dev_stress_invar2, 
                             50,
                             cmap="plasma", origin="lower")
    ax.set_xlabel("x coordinate [m]", fontsize=14)
    ax.set_title(r"$\sigma'_{II}$ [Pa]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
        
    ax = axes[2, 2]
    contour = ax.tricontourf(quad_coord[0, :], quad_coord[1, :], quad_dense, 
                             50, cmap="Spectral", origin="lower")
    ax.set_xlabel("x coordinate [m]", fontsize=14)
    ax.set_title(r"$\rho$ [kg/m$^3$]", fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    tcurrent = t_steps[-1] + dt
    t_steps.append(tcurrent)
    fig.suptitle("Time: {:.2f}".format(tcurrent))
    
    fig_flow.clear()
    ax = fig_flow.subplots(nrows=1, ncols=1)
    contour = ax.tricontourf(quad_coord[0, :], quad_coord[1, :], quad_dense, 
                             50, cmap="Spectral", origin="lower")
    ax.quiver(gcoord_v[0, :], gcoord_v[1, :], vx, vy, color='w')
    ax.set_xlabel("x coordinate [m]", fontsize=14)
    ax.set_xlabel("y coordinate [m]", fontsize=14)
    ax.set_title("Time: {:.2f}".format(tcurrent), fontsize=16)
    plt.colorbar(mappable=contour, ax=ax)
    ax.grid(which="both")
    ax.axis("equal")
    
    if output:
        fig.savefig("./output/intrusion_{}.png".format(i_time), format="png", bbox_inches="tight", dpi=150)
        fig_flow.savefig("./output/intrusion_flow_{}.png".format(i_time), format="png", bbox_inches="tight", dpi=150)
    plt.pause(0.1)
    
    if not dynamic:
        break
    
    """Advection"""
    gcoord_v[0, :] += dt*vx
    gcoord_v[1, :] += dt*vy
    gcoord_p = gcoord_v[:, node_idx_p]

plt.show()

