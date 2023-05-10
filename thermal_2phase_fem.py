"""
1D model for calculating eigenmodes of torsional Alfven waves
FEM implementation
"""

import numpy as np
import os
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem import mesh, quad
import h5py


"""IO Setting"""

output = True
# output_dir = "./output/eigenmodes_1D/eigenmodes_Pm0_Lu2000_freeslip/"
autogen_output_dir = True

# save_pattern_fname = "./output/specspy_TO_noslip_cheby500"
save_pattern_fname = None

"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_2phase as cfg

if autogen_output_dir:
    output_dir = os.path.join("./output/thermal_2phase_1D", "T_evolution")


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_2phase as cfg


"""Mesh generation"""

# Maximum element size
ds_max = 0.01
# Collocation: the meshes are of the same type and overlap
elem = mesh.QuadraticElement()
# Quadrature rules
xi_quad, wt_quad = quad.gaussian_quad(5)
n_quad = wt_quad.size

# Number of elements
n_elem = int(np.ceil(Ls/ds_max))
# Number of total nodes (multiple nodes per element depending on the order)
n_node = n_elem*(elem.n_nodes - 1) + 1
# Number of degrees of freedom per node (two fields, u and b)
n_dof_node = 2
# Number of degrees of freedom
n_dof = n_dof_node*n_node
# Spacial spacing
ds = Ls/n_elem

# Node coordinates
gcoord_node = np.linspace(*s_range, num=n_node)

# Connectivity matrices
elem2node = np.array([(elem.n_nodes - 1)*i for i in range(n_elem)])
elem2node = np.stack([elem2node + i for i in range(elem.n_nodes)], axis=-1)
node2dof_u = np.array([n_dof_node*i for i in range(n_node)])
node2dof_b = np.array([n_dof_node*i + 1 for i in range(n_node)])
elem2dof_u = np.array([node2dof_u[elem2node[i]].flatten() for i in range(n_elem)])
elem2dof_b = np.array([node2dof_b[elem2node[i]].flatten() for i in range(n_elem)])


"""Time-stepping setup"""
tmax = 1.
dt = 1e-3
step_verbose = 100


"""Matrix assembly
Since this is an Eulerian mesh with time-independent coefficients, 
the global matrices remain unchanged and can be reused
"""

# Pre-evaluate shape functions at integration points
N_vals = elem.f_shape(xi_quad)
dN_vals = elem.f_dshape(xi_quad)

# Pre-allocate global matrices ----- historical dense formulation
# K_mat = np.zeros((n_dof, n_dof))
# M_mat = np.zeros((n_dof, n_dof))

# Sparse formulation
stride = elem.n_nodes**2
K_vals = np.zeros(4*n_elem*stride)
K_rows = np.zeros(4*n_elem*stride, dtype=np.int32)
K_cols = np.zeros(4*n_elem*stride, dtype=np.int32)

M_vals = np.zeros(2*n_elem*stride)
M_rows = np.zeros(2*n_elem*stride, dtype=np.int32)
M_cols = np.zeros(2*n_elem*stride, dtype=np.int32)

for i_elem in range(n_elem):
    
    # Current dofs - used to determine where the submatrix is added
    dof_temp_u = elem2dof_u[i_elem]
    dof_temp_b = elem2dof_b[i_elem]
    # Current coordinates of nodes
    coord_temp = gcoord_node[elem2node[i_elem]]
    # Integration points needed for the element
    pt_quad = coord_temp @ N_vals
    # Jacobians
    jac_temp = coord_temp @ dN_vals
    # dN/ds at integration points
    dN_ds = dN_vals / jac_temp
    
    # Kuu submatrix
    Kuu0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kuu0 = np.sum(wt_quad*jac_temp*cfg.k_uu0(pt_quad)*Kuu0, axis=-1)
    Kuu1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kuu1 = np.sum(wt_quad*jac_temp*(cfg.k_uu1(pt_quad) - cfg.dk_uu2(pt_quad))*Kuu1, axis=-1)
    Kuu2 = np.stack([np.outer(dN_ds[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kuu2 = np.sum(-wt_quad*jac_temp*cfg.k_uu2(pt_quad)*Kuu2, axis=-1)
    
    # K_mat[np.ix_(dof_temp_u, dof_temp_u)] += Kuu1 + Kuu2
    iptr_slice = slice((4*i_elem)*stride, (4*i_elem + 1)*stride)
    K_vals[iptr_slice] = (Kuu0 + Kuu1 + Kuu2).flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_u, dof_temp_u, indexing='ij')
    K_rows[iptr_slice] = i_rows.flatten()
    K_cols[iptr_slice] = i_cols.flatten()
    
    # Kub submatrix
    Kub1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kub1 = np.sum(wt_quad*jac_temp*cfg.k_ub1(pt_quad)*Kub1, axis=-1)
    Kub0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kub0 = np.sum(wt_quad*jac_temp*cfg.k_ub0(pt_quad)*Kub0, axis=-1)
    
    # K_mat[np.ix_(dof_temp_u, dof_temp_b)] += Kub1 + Kub0
    iptr_slice = slice((4*i_elem + 1)*stride, (4*i_elem + 2)*stride)
    K_vals[iptr_slice] = (Kub1 + Kub0).flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_u, dof_temp_b, indexing='ij')
    K_rows[iptr_slice] = i_rows.flatten()
    K_cols[iptr_slice] = i_cols.flatten()
    
    # Mu submatrix
    Mu = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Mu = np.sum(wt_quad*jac_temp*cfg.m_u(pt_quad)*Mu, axis=-1)
    
    # M_mat[np.ix_(dof_temp_u, dof_temp_u)] += Mu
    iptr_slice = slice((2*i_elem)*stride, (2*i_elem + 1)*stride)
    M_vals[iptr_slice] = Mu.flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_u, dof_temp_u, indexing='ij')
    M_rows[iptr_slice] = i_rows.flatten()
    M_cols[iptr_slice] = i_cols.flatten()
    
    # Kbu submatrix
    Kbu1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbu1 = np.sum(wt_quad*jac_temp*cfg.k_bu1(pt_quad)*Kbu1, axis=-1)
    Kbu0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kbu0 = np.sum(wt_quad*jac_temp*cfg.k_bu0(pt_quad)*Kbu0, axis=-1)
    
    # K_mat[np.ix_(dof_temp_b, dof_temp_u)] += Kbu
    iptr_slice = slice((4*i_elem + 2)*stride, (4*i_elem + 3)*stride)
    K_vals[iptr_slice] = (Kbu0 + Kbu1).flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_b, dof_temp_u, indexing='ij')
    K_rows[iptr_slice] = i_rows.flatten()
    K_cols[iptr_slice] = i_cols.flatten()
    
    # Kbb submatrix
    Kbb2 = np.stack([np.outer(dN_ds[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb2 = np.sum(-wt_quad*jac_temp*cfg.k_bb2(pt_quad)*Kbb2, axis=-1)
    Kbb1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb1 = np.sum(wt_quad*jac_temp*(cfg.k_bb1(pt_quad) - cfg.dk_bb2(pt_quad))*Kbb1, axis=-1)
    Kbb0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kbb0 = np.sum(wt_quad*jac_temp*cfg.k_bb0(pt_quad)*Kbb0, axis=-1)
    
    # K_mat[np.ix_(dof_temp_b, dof_temp_b)] += Kbb2 + Kbb1 + Kbb0
    iptr_slice = slice((4*i_elem + 3)*stride, (4*i_elem + 4)*stride)
    K_vals[iptr_slice] = (Kbb2 + Kbb1 + Kbb0).flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_b, dof_temp_b, indexing='ij')
    K_rows[iptr_slice] = i_rows.flatten()
    K_cols[iptr_slice] = i_cols.flatten()
    
    # Mb submatrix
    Mb = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Mb = np.sum(wt_quad*jac_temp*cfg.m_b(pt_quad)*Mb, axis=-1)
    
    # M_mat[np.ix_(dof_temp_b, dof_temp_b)] += Mb
    iptr_slice = slice((2*i_elem + 1)*stride, (2*i_elem + 2)*stride)
    M_vals[iptr_slice] = Mb.flatten()
    i_rows, i_cols = np.meshgrid(dof_temp_b, dof_temp_b, indexing='ij')
    M_rows[iptr_slice] = i_rows.flatten()
    M_cols[iptr_slice] = i_cols.flatten()

# Assemble sparse matrices
M_mat = sparse.coo_array((M_vals, (M_rows, M_cols)), shape=(n_dof, n_dof)).tocsr()
K_mat = sparse.coo_array((K_vals, (K_rows, K_cols)), shape=(n_dof, n_dof)).tocsr()
print("Matrices assembled")

import matplotlib.pyplot as plt

# Sparse pattern plot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax = axes[0]
ax.spy(K_mat, markersize=1.)
ax.set_title("K")
ax = axes[1]
ax.spy(M_mat, markersize=1.)
ax.set_title("M")

if save_pattern_fname is not None:
    plt.savefig(save_pattern_fname + '.png', format='png', dpi=150, bbox_inches="tight")
plt.show()



"""Time stepping"""

t_array = np.arange(0, tmax, dt)
solution_array = np.zeros((t_array.size, n_dof))

# Initial condition
solution_array[0, :] = 1.

for t_id in range(t_array.size - 1):
    L_mat = M_mat - dt*K_mat
    f_vec = M_mat @ solution_array[t_id]
    
    # Boundary conditions
    Tleft = cfg.bound_left(t_array[t_id+1])
    solution_array[t_id+1, 0] = Tleft
    f_vec -= L_mat @ solution_array[t_id+1, :]
    L_mat = L_mat[1:, 1:]
    f_vec = f_vec[1:]
        
    # print(f_vec)
    
    solution_step = splinalg.spsolve(L_mat, f_vec)
    solution_array[t_id+1, 1:] = solution_step
    
    if (t_id + 1) % step_verbose == 0:
        print("{:d}/{:d} steps solved.".format(t_id + 1, t_array.size))


"""Output"""

import h5py

# np.save("./output/K_mat_vectorized_v3.npy", arr=K_mat)
# np.save("./output/M_mat_vectorized_v3.npy", arr=M_mat)

if output:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_fname = "solution_fem{:d}.h5".format(n_elem)
    with h5py.File(os.path.join(output_dir, save_fname), 'x') as f_write:
        f_write.create_dataset("solution", data=solution_array)
        f_write.create_dataset("mesh", data=gcoord_node)
    print("Solutions saved to " + output_dir)
