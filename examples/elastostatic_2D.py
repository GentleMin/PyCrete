# -*- coding: utf-8 -*-
"""
FEM code for 2D elastostatics
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from fem import mesh, quad


"""Physics setup"""

# Domain
x_range = [0, 10]
y_range = [0, 2]
Lx = x_range[1] - x_range[0]
Ly = y_range[1] - y_range[0]

output = False

def iso_elasticity(x, y, nu=0.25, E=1e+8):
    """Isotropic elasticity matrix"""
    assert x.shape == y.shape
    D = E/(1 + nu)/(1 - 2*nu)*np.array([[1 - nu, nu, 0],
                                        [nu, 1 - nu, 0],
                                        [0, 0, (1 - 2*nu)/2]])
    return np.multiply.outer(np.ones(x.shape), D)

def calc_traction(x, y):
    """Traction
    """
    assert x.shape == y.shape
    return np.multiply.outer(np.ones(x.shape), np.array([0, 0]))

def calc_body_force(x, y):
    """Body force
    """
    assert x.shape == y.shape
    return np.multiply.outer(np.ones(x.shape), 1*np.array([0, -1e+4]))

def get_rectangular_boundaries(x, y, x_range, y_range):
    """Find boundaries according to a rectangular domain
    """
    bc_idx_top = np.arange(y.size)[np.abs(y - y_range[1]) <= 1e-6]
    bc_idx_btm = np.arange(y.size)[np.abs(y - y_range[0]) <= 1e-6]
    bc_idx_left = np.arange(x.size)[np.abs(x - x_range[0]) <= 1e-6]
    bc_idx_right = np.arange(x.size)[np.abs(x - x_range[1]) <= 1e-6]
    return bc_idx_top, bc_idx_btm, bc_idx_left, bc_idx_right
    
def dirichlet_bc(x, y, x_range, y_range):
    """Generate Dirichlet BC
    """
    # Predefined scheme 1: Encastre bottom (=> ux = uy = 0)
    # _, bc_idx_btm, _, _ = get_rectangular_boundaries(x, y, x_range, y_range)
    # bc_dof = np.concatenate([node_dof_map[bc_idx_btm, 1], node_dof_map[bc_idx_btm, 0]])
    # bc_val = np.zeros(bc_dof.shape)
    
    # Predefined scheme 2: No-penetration bottom (=> uy = 0)
    # _, bc_idx_btm, _, _ = get_rectangular_boundaries(x, y, x_range, y_range)    
    # bc_dof = np.concatenate([node_dof_map[bc_idx_btm, 1], np.atleast_1d(node_dof_map[bc_idx_btm[bc_idx_btm.size // 2], 0])])
    # bc_val = np.zeros(bc_dof.shape)
    
    # Predefined scheme 3: Pure shear
    # _, bc_idx_btm, bc_idx_left, bc_idx_right = get_rectangular_boundaries(x, y, x_range, y_range)
    # bc_dof = np.concatenate([node_dof_map[bc_idx_btm, 1], node_dof_map[bc_idx_left, 0], node_dof_map[bc_idx_right, 0]])
    # bc_val = np.concatenate([np.zeros(bc_idx_btm.size), 0.05*y[bc_idx_left], 0.05*y[bc_idx_right]])
    # bc_dof = np.concatenate([node_dof_map[bc_idx_btm, 1], node_dof_map[bc_idx_left, 0]])
    # bc_val = np.concatenate([np.zeros(bc_idx_btm.size), 0.05*y[bc_idx_left]])
    # bc_dof = np.concatenate([node_dof_map[bc_idx_left, 1], node_dof_map[bc_idx_right, 1],
    #                          node_dof_map[bc_idx_left, 0], node_dof_map[bc_idx_right, 0]])
    # bc_val = np.concatenate([np.zeros(bc_idx_left.size), np.zeros(bc_idx_right.size), 
    #                          0.05*y[bc_idx_left], 0.05*y[bc_idx_right]])
    
    # Predefined scheme 4: beam bending under gravity with encastre left boundary
    _, _, bc_idx_left, _ = get_rectangular_boundaries(x, y, x_range, y_range)
    bc_dof = np.concatenate([node_dof_map[bc_idx_left, 0], node_dof_map[bc_idx_left, 1]])
    bc_val = np.concatenate([np.zeros(bc_idx_left.size), np.zeros(bc_idx_left.size)])
    
    return bc_dof, bc_val


"""Mesh setup"""
n_elem_x = 10               # Number of elements in x dir
n_elem_y = 2
n_elem_tot = n_elem_y*n_elem_x
dx = Lx/n_elem_x            # Element dimensions
dy = Ly/n_elem_y

ndof_node = 2

# Quadrature configuration
xi_quad, wt_quad = quad.quad_2d["9-pt"]
print("Using 9-pt quadrature.")
n_quad = wt_quad.size

# Mesh-related config
element = mesh.BiQuadraticElement()
print("Using {}.".format(type(element).__name__))
nodes_elem = element.n_nodes
order_elem = element.order

if isinstance(element, mesh.BiLinearElement):
    n_node_x = n_elem_x + 1
    n_node_y = n_elem_y + 1
    n_node_tot = n_node_y*n_node_x
    
    gcoord_x = np.linspace(*x_range, num=n_node_x)
    gcoord_y = np.linspace(*y_range, num=n_node_y)
    gcoord_X, gcoord_Y = np.meshgrid(gcoord_x, gcoord_y, indexing='xy')
    gcoord_nodes = np.stack([gcoord_X.flatten(), gcoord_Y.flatten()], axis=0)
    elem_node_map = np.array([[n_node_x*i + j, n_node_x*(i+1) + j, n_node_x*(i+1) + (j+1), n_node_x*i + (j+1)] 
                              for i in range(n_elem_y) for j in range(n_elem_x)])
    
elif isinstance(element, mesh.BiQuadraticElement):
    n_node_x = 2*n_elem_x + 1
    n_node_y = 2*n_elem_y + 1
    n_node_tot = n_node_x*n_node_y
    
    gcoord_x = np.linspace(*x_range, num=n_node_x)
    gcoord_y = np.linspace(*y_range, num=n_node_y)
    gcoord_X, gcoord_Y = np.meshgrid(gcoord_x, gcoord_y, indexing='xy')
    gcoord_nodes = np.stack([gcoord_X.flatten(), gcoord_Y.flatten()], axis=0)
    elem_node_map = np.array([[n_node_x*2*i + 2*j, n_node_x*2*(i+1) + 2*j, n_node_x*2*(i+1) + 2*(j+1), n_node_x*2*i + 2*(j+1), 
                               n_node_x*(2*i+1) + 2*j, n_node_x*2*(i+1) + 2*j + 1, n_node_x*(2*i+1) + 2*(j+1), n_node_x*2*i + 2*j + 1, 
                               n_node_x*(2*i+1) + 2*j + 1] for i in range(n_elem_y) for j in range(n_elem_x)])

n_dof_tot = 2*n_node_tot
node_dof_map = np.stack([2*np.arange(n_node_tot), 2*np.arange(n_node_tot) + 1], axis=-1)
elem_dof_map = np.array([node_dof_map[elem_nodes, :].flatten() for elem_nodes in elem_node_map])


"""Initialization"""
bc_dof, bc_val = dirichlet_bc(gcoord_nodes[0, :], gcoord_nodes[1, :], x_range, y_range)
# np.random.seed(1105)
# gcoord_nodes[1, bc_dof[-n_node_x:]] += 2*np.random.rand(n_node_x)
# gcoord_nodes[1, bc_dof[n_node_x:2*n_node_x]] += 10*gcoord_x*(Lx - gcoord_x)/Lx**2

K_global = np.zeros((n_dof_tot, n_dof_tot))
F_global = np.zeros(n_dof_tot)

N_vals = element.f_shape(xi_quad)
N_mat = np.concatenate([np.array([[N_vals[i], np.zeros(N_vals[i].shape)], 
                                  [np.zeros(N_vals[i].shape), N_vals[i]]]) for i in range(nodes_elem)], axis=1)
dN_vals = element.f_dshape(xi_quad)


"""Assembling global matrices"""
for i_elem in range(n_elem_tot):
    
    nodes_temp = elem_node_map[i_elem]
    coord_temp = gcoord_nodes[:, nodes_temp]
    dof_temp = elem_dof_map[i_elem]
    
    pt_quad = coord_temp @ N_vals
    
    elasticity = iso_elasticity(pt_quad[0, :], pt_quad[1, :])
    body_force = calc_body_force(pt_quad[0, :], pt_quad[1, :])
    
    jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
    jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
    dN_dx = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1).transpose((1, 0, 2))
    B_mat = np.concatenate([np.array([[dN_dx[i, 0], np.zeros(n_quad)], 
                                      [np.zeros(n_quad), dN_dx[i, 1]],
                                      [dN_dx[i, 1], dN_dx[i, 0]]]) for i in range(nodes_elem)], axis=1)
    
    K_local = np.stack([B_mat[:, :, i].T @ elasticity[i, :, :] @ B_mat[:, :, i] for i in range(n_quad)], axis=-1)
    K_local = np.sum(wt_quad*K_local*jac_det, axis=-1)
    
    # print(body_force)
    F_local = np.stack([N_mat[:, :, i].T @ body_force[i] for i in range(n_quad)], axis=-1)
    F_local = np.sum(wt_quad*F_local*jac_det, axis=-1)
    
    idx = np.ix_(dof_temp, dof_temp)
    K_global[idx] += K_local
    F_global[dof_temp] += F_local

# print(elasticity[0, :, :])
L = K_global
b = F_global

# Apply BC (reduced matrix version)
b -= L[:, bc_dof] @ np.asarray(bc_val)
rem_idx = [idx for idx in range(n_dof_tot) if idx not in bc_dof]
b = b[rem_idx]
L = L[np.ix_(rem_idx, rem_idx)]

u_global = np.zeros(n_dof_tot)
print("cond(L) = {:.2e}".format(np.linalg.cond(L)))
u_global[rem_idx] = np.linalg.solve(L, b)
u_global[bc_dof] = np.asarray(bc_val)

# L[bc_dof, :] = 0
# L[np.ix_(bc_dof, bc_dof)] = np.eye(bc_dof.size)
# b[bc_dof] = bc_val
# print("cond(L) = {:.2e}".format(np.linalg.cond(L)))
# u_global = np.linalg.solve(L, b)


"""Unravel fields and visualize"""

u_x = u_global[node_dof_map[:, 0]]
u_y = u_global[node_dof_map[:, 1]]

# print(u_x.min(), u_x.max(), u_y.min(), u_y.max())

quad_coord = np.zeros((2, n_elem_tot*n_quad))
quad_disp = np.zeros((2, n_elem_tot*n_quad))
quad_stress = np.zeros((3, n_elem_tot*n_quad))
quad_strain = np.zeros((3, n_elem_tot*n_quad))

for i_elem in range(n_elem_tot):
    
    nodes_temp = elem_node_map[i_elem]
    coord_temp = gcoord_nodes[:, nodes_temp]
    dof_temp = elem_dof_map[i_elem]
    
    pt_quad = coord_temp @ N_vals
    quad_coord[:, i_elem*n_quad:(i_elem+1)*n_quad] = pt_quad
    
    u_nodes = np.array([u_x[nodes_temp], u_y[nodes_temp]]).flatten(order='F')
    # print(u_nodes)
    quad_disp[0, i_elem*n_quad:(i_elem+1)*n_quad] = u_x[nodes_temp] @ N_vals
    quad_disp[1, i_elem*n_quad:(i_elem+1)*n_quad] = u_y[nodes_temp] @ N_vals
    elasticity = iso_elasticity(pt_quad[0, :], pt_quad[1, :])
    body_force = calc_body_force(pt_quad[0, :], pt_quad[1, :])
    
    jac_temp = np.stack([dN_vals[:, :, i].T @ coord_temp.T for i in range(n_quad)], axis=-1)
    jac_det = np.array([np.linalg.det(jac_temp[:, :, i]) for i in range(n_quad)])
    dN_dx = np.stack([np.linalg.solve(jac_temp[:, :, i], dN_vals[:, :, i].T) for i in range(n_quad)], axis=-1).transpose((1, 0, 2))
    B_mat = np.concatenate([np.array([[dN_dx[i, 0], np.zeros(n_quad)], 
                                      [np.zeros(n_quad), dN_dx[i, 1]],
                                      [dN_dx[i, 1], dN_dx[i, 0]]]) for i in range(nodes_elem)], axis=1)
        
    strain_temp = np.stack([B_mat[:, :, i] @ u_nodes for i in range(n_quad)], axis=-1)
    stress_temp = np.stack([elasticity[i, :, :] @ strain_temp[:, i] for i in range(n_quad)], axis=-1)
    # print("Element {}: exx =".format(i_elem), strain_temp[0, :])
    # print("Element {}: eyy =".format(i_elem), strain_temp[1, :])
    # print("Element {}: exy =".format(i_elem), strain_temp[2, :])
    # print("Element {}: sxx =".format(i_elem), stress_temp[0, :])
    # print("Element {}: syy =".format(i_elem), stress_temp[1, :])
    # print("Element {}: sxy =".format(i_elem), stress_temp[2, :])
    quad_strain[:, i_elem*n_quad:(i_elem+1)*n_quad] = strain_temp
    quad_stress[:, i_elem*n_quad:(i_elem+1)*n_quad] = stress_temp

dev_stress_invar2 = np.sqrt(quad_stress[2, :]**2 + (quad_stress[0, :] - quad_stress[1, :])**2/4)


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

ax = axes[0, 0]
contour = ax.tricontourf(gcoord_nodes[0, :] + u_x, gcoord_nodes[1, :] + u_y, u_x, 
                         50, vmin=-np.abs(u_x).max(), vmax=np.abs(u_x).max(),
                        cmap="coolwarm", origin="lower")
ax.set_ylabel("y coordinate [m]", fontsize=14)
ax.set_title(r"$u_x$ [m]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

ax = axes[0, 1]
contour = ax.tricontourf(gcoord_nodes[0, :] + u_x, gcoord_nodes[1, :] + u_y, u_y, 
                         50, vmin=-np.abs(u_y).max(), vmax=np.abs(u_y).max(),
                         cmap="coolwarm", origin="lower")
# plt.imshow(np.reshape(u_y, (n_node_y, n_node_x)))
ax.set_title(r"$u_y$ [m]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

ax = axes[0, 2]
contour = ax.tricontourf(quad_coord[0, :] + quad_disp[0, :], 
                         quad_coord[1, :] + quad_disp[1, :], 
                         dev_stress_invar2, 
                         50,
                        #  np.linspace(dev_stress_invar2.min(), dev_stress_invar2.max(), 50), 
                         cmap="plasma", origin="lower")
# plt.imshow(np.reshape(u_y, (n_node_y, n_node_x)))
ax.set_title(r"$\sigma'_{II}$ [Pa]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

ax = axes[1, 0]
contour = ax.tricontourf(quad_coord[0, :] + quad_disp[0, :], 
                         quad_coord[1, :] + quad_disp[1, :], 
                         quad_stress[0, :], 
                         50, 
                         vmin=-np.abs(quad_stress[0, :]).max(), vmax = np.abs(quad_stress[0, :]).max(),
                        #  np.linspace(quad_stress[0, :].min(), quad_stress[0, :].max(), 50), 
                         cmap="Spectral", origin="lower")
# plt.imshow(np.reshape(u_y, (n_node_y, n_node_x)))
ax.set_xlabel("x coordinate [m]", fontsize=14)
ax.set_ylabel("y coordinate [m]", fontsize=14)
ax.set_title(r"$\sigma_{xx}$ [Pa]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

ax = axes[1, 1]
contour = ax.tricontourf(quad_coord[0, :] + quad_disp[0, :], 
                         quad_coord[1, :] + quad_disp[1, :], 
                         quad_stress[1, :], 
                         50, 
                         vmin=-np.abs(quad_stress[1, :]).max(), vmax = np.abs(quad_stress[1, :]).max(),
                        #  np.linspace(-np.abs(quad_stress[1, :]).max(), np.abs(quad_stress[1, :]).max(), 50), 
                         cmap="Spectral", origin="lower")
# plt.imshow(np.reshape(u_y, (n_node_y, n_node_x)))
ax.set_xlabel("x coordinate [m]", fontsize=14)
ax.set_title(r"$\sigma_{yy}$ [Pa]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

ax = axes[1, 2]
contour = ax.tricontourf(quad_coord[0, :] + quad_disp[0, :], 
                         quad_coord[1, :] + quad_disp[1, :], 
                         quad_stress[2, :], 
                         50,
                         cmap="Spectral", origin="lower")
# plt.imshow(np.reshape(u_y, (n_node_y, n_node_x)))
ax.set_xlabel("x coordinate [m]", fontsize=14)
ax.set_title(r"$\gamma_{xy}$ [Pa]", fontsize=16)
plt.colorbar(mappable=contour, ax=ax)
ax.grid(which="both")
ax.axis("equal")

if output:
    plt.savefig("./output/beam_bending.png", format="png", bbox_inches="tight", dpi=300)
plt.show()


