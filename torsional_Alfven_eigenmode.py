"""
1D model for calculating eigenmodes of torsional Alfven waves
FEM implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
from fem import mesh, quad
import h5py


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]

# Magnetic Prandtl number
Pm = 0
# Lundquist number
Lu = 2000
# Radial magnetic field profile
def Bs2_S1(s):
    return 63/184/np.pi*s**2*(1 - s**2)

def Bs2_S2(s):
    return 3/(28*182*16*np.pi)*s**2*(191222 - 734738*s**2 + 1060347*s**4 - 680108*s**6 + 163592*s**8)


"""Introducing coefficients"""

def k_uu2(s):
    return Pm/Lu*s**2*(1 - s**2)

def k_uu1(s):
    return 3*Pm/Lu*s*(1 - s**2)

def k_ub1(s):
    return s*(1 - s**2)
    
def k_ub0(s):
    return 2 - s**2

def k_bu1(s, Bs2):
    return s**2*Bs2(s)

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

"""Test case: standard wave equation"""

def k_uu2(s):
    return np.zeros(s.shape)

def k_uu1(s):
    return np.zeros(s.shape)

def k_ub1(s):
    return np.ones(s.shape)
    
def k_ub0(s):
    return np.zeros(s.shape)

def k_bu1(s, Bs2):
    return np.ones(s.shape)

def k_bb2(s):
    return np.zeros(s.shape)

def k_bb1(s):
    return np.zeros(s.shape)

def k_bb0(s):
    return np.zeros(s.shape)

# Necessary derivatives

def dk_uu2(s):
    return np.zeros(s.shape)

def dk_bb2(s):
    return np.zeros(s.shape)


"""Mesh generation"""

# Maximum element size
ds_max = 0.02
# Collocation: the meshes are of the same type and overlap
elem = mesh.QuadraticElement()
# Quadrature rules
xi_quad, wt_quad = quad.quad_1d["5-pt"]
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
    Kuu1 = np.sum(wt_quad*jac_temp*(k_uu1(pt_quad) - dk_uu2(pt_quad))*Kuu1, axis=-1)
    Kuu2 = np.stack([np.outer(dN_ds[:, 1], dN_ds[:, 1]) for i in range(n_quad)], axis=-1)
    Kuu2 = np.sum(-wt_quad*jac_temp*k_uu2(pt_quad)*Kuu2, axis=-1)
    K_mat[np.ix_(dof_temp_u, dof_temp_u)] += Kuu1 + Kuu2
    
    # Kub submatrix
    Kub1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kub1 = np.sum(wt_quad*jac_temp*k_ub1(pt_quad)*Kub1, axis=-1)
    Kub0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kub0 = np.sum(wt_quad*jac_temp*k_ub0(pt_quad)*Kub0, axis=-1)
    K_mat[np.ix_(dof_temp_u, dof_temp_b)] += Kub1 + Kub0
    
    # Mu submatrix
    Mu = np.sum(wt_quad*jac_temp*np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1), axis=-1)
    M_mat[np.ix_(dof_temp_u, dof_temp_u)] += Mu
    
    # Kbu submatrix
    Kbu = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbu = np.sum(wt_quad*jac_temp*k_bu1(pt_quad, Bs2_S1)*Kbu, axis=-1)
    K_mat[np.ix_(dof_temp_b, dof_temp_u)] += Kbu
    
    # Kbb submatrix
    Kbb2 = np.stack([np.outer(dN_ds[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb2 = np.sum(-wt_quad*jac_temp*k_bb2(pt_quad)*Kbb2, axis=-1)
    Kbb1 = np.stack([np.outer(N_vals[:, i], dN_ds[:, i]) for i in range(n_quad)], axis=-1)
    Kbb1 = np.sum(wt_quad*jac_temp*(k_bb1(pt_quad) - dk_bb2(pt_quad))*Kbb1, axis=-1)
    Kbb0 = np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1)
    Kbb0 = np.sum(wt_quad*jac_temp*k_bb0(pt_quad)*Kbb0, axis=-1)
    K_mat[np.ix_(dof_temp_b, dof_temp_b)] += Kbb2 + Kbb1 + Kbb0
    
    # Mb submatrix
    Mb = np.sum(wt_quad*jac_temp*np.stack([np.outer(N_vals[:, i], N_vals[:, i]) for i in range(n_quad)], axis=-1), axis=-1)
    M_mat[np.ix_(dof_temp_b, dof_temp_b)] += Mb


"""Applying Boundary conditions"""

# K_mat[[1, -1], :] = 0
# K_mat[:, [1, -1]] = 0
# K_mat[np.ix_([1, -1], [1, -1])] = np.eye(2)

idx_rem = [i for i in range(n_dof) if i not in (1, n_dof - 1)]
M_mat = M_mat[np.ix_(idx_rem, idx_rem)]
K_mat = K_mat[np.ix_(idx_rem, idx_rem)]

print(np.linalg.cond(K_mat), np.linalg.cond(M_mat))

"""Solving the eigenvalue problem"""
# w, v = splinalg.eigs(sparse.csc_array(K_mat), k=6, M=sparse.csc_array(M_mat), which="LR", return_eigenvectors=True, tol=1e-3, maxiter=10000)
w, v = linalg.eig(K_mat, M_mat)
# print(w)

with h5py.File("./output/eigenmodes_1Dwave_n50.h5", 'x') as f_write:
    f_write.create_dataset("nodes", data=gcoord_node)
    f_write.create_dataset("cond", data=np.array([np.linalg.cond(K_mat), np.linalg.cond(M_mat)]))
    idx_sort = np.argsort(w)
    w_sort = w[idx_sort]
    v_sort = np.zeros((n_dof, n_dof - 2), dtype=np.complex128)
    v_sort[idx_rem, :] = v[:, idx_sort]
    f_write.create_dataset("eigenvals", data=w_sort)
    f_write.create_dataset("eigenfuns", data=v_sort)


"""Matrices visualization"""

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

ax = axes[0]
ax.spy(M_mat)
ax.set_title("M")

ax = axes[1]
ax.spy(K_mat)
ax.set_title("K")

plt.savefig("./output/matrix_sparsity.png", format="png", dpi=150, bbox_inches="tight")
plt.show()

