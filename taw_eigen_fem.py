"""
1D model for calculating eigenmodes of torsional Alfven waves
FEM implementation
"""

import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem import mesh, quad
import h5py


"""IO Setting"""

save_eigen = True
save_eigen_fname = "./output/eigenmodes_Pm0"

sparse_solver = True


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_TO as cfg


"""Mesh generation"""

# Maximum element size
ds_max = 0.02
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


"""Matrix assembly
Since this is an Eulerian mesh with time-independent coefficients, 
the global matrices remain unchanged and can be reused
"""

# Pre-evaluate shape functions at integration points
N_vals = elem.f_shape(xi_quad)
dN_vals = elem.f_dshape(xi_quad)

# Pre-allocate global matrices -----> convert to fully sparse formulation in the future
K_mat = np.zeros((n_dof, n_dof))
M_mat = np.zeros((n_dof, n_dof))

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
    Kuu1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kuu1 = np.sum(wt_quad*jac_temp*(cfg.k_uu1(pt_quad) - cfg.dk_uu2(pt_quad))*Kuu1, axis=-1)
    Kuu2 = np.stack([np.outer(dN_ds[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kuu2 = np.sum(-wt_quad*jac_temp*cfg.k_uu2(pt_quad)*Kuu2, axis=-1)
    K_mat[np.ix_(dof_temp_u, dof_temp_u)] += Kuu1 + Kuu2
    
    # Kub submatrix
    Kub1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kub1 = np.sum(wt_quad*jac_temp*cfg.k_ub1(pt_quad)*Kub1, axis=-1)
    Kub0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kub0 = np.sum(wt_quad*jac_temp*cfg.k_ub0(pt_quad)*Kub0, axis=-1)
    K_mat[np.ix_(dof_temp_u, dof_temp_b)] += Kub1 + Kub0
    
    # Mu submatrix
    Mu = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Mu = np.sum(wt_quad*jac_temp*cfg.m_u(pt_quad)*Mu, axis=-1)
    M_mat[np.ix_(dof_temp_u, dof_temp_u)] += Mu
    
    # Kbu submatrix
    Kbu = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbu = np.sum(wt_quad*jac_temp*cfg.k_bu1(pt_quad)*Kbu, axis=-1)
    K_mat[np.ix_(dof_temp_b, dof_temp_u)] += Kbu
    
    # Kbb submatrix
    Kbb2 = np.stack([np.outer(dN_ds[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb2 = np.sum(-wt_quad*jac_temp*cfg.k_bb2(pt_quad)*Kbb2, axis=-1)
    Kbb1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb1 = np.sum(wt_quad*jac_temp*(cfg.k_bb1(pt_quad) - cfg.dk_bb2(pt_quad))*Kbb1, axis=-1)
    Kbb0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kbb0 = np.sum(wt_quad*jac_temp*cfg.k_bb0(pt_quad)*Kbb0, axis=-1)
    K_mat[np.ix_(dof_temp_b, dof_temp_b)] += Kbb2 + Kbb1 + Kbb0
    
    # Mb submatrix
    Mb = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Mb = np.sum(wt_quad*jac_temp*cfg.m_b(pt_quad)*Mb, axis=-1)
    M_mat[np.ix_(dof_temp_b, dof_temp_b)] += Mb


# print(Kuu2 + Kuu1)
# print(Kub1 + Kub0)
# print(Kbu)
# print(Kbb2 + Kbb1 + Kbb0)
# print(Mu)
# print(Mb)

"""Applying homogeneous Dirichlet boundary conditions"""

idx_rem = [i for i in range(n_dof) if i not in (1, n_dof - 1)]
M_mat = M_mat[np.ix_(idx_rem, idx_rem)]
K_mat = K_mat[np.ix_(idx_rem, idx_rem)]

print(np.linalg.cond(K_mat), np.linalg.cond(M_mat))

"""Solving the eigenvalue problem"""
if sparse_solver:
    w, v = splinalg.eigs(sparse.csc_array(K_mat), k=70, M=sparse.csc_array(M_mat), which="LR", return_eigenvectors=True, tol=1e-3, maxiter=1000)
else:
    w, v = linalg.eig(K_mat, M_mat)

if save_eigen:
    with h5py.File(save_eigen_fname + ".h5", 'x') as f_write:
        f_write.create_dataset("nodes", data=gcoord_node)
        f_write.create_dataset("cond", data=np.array([np.linalg.cond(K_mat), np.linalg.cond(M_mat)]))
        idx_sort = np.argsort(w)
        w_sort = w[idx_sort]
        v_sort = np.zeros((n_dof, v.shape[1]), dtype=np.complex128)
        v_sort[idx_rem, :] = v[:, idx_sort]
        f_write.create_dataset("eigenvals", data=w_sort)
        f_write.create_dataset("eigenfuns", data=v_sort)

