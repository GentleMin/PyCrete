"""
1D model for calculating eigenmodes of torsional Alfven waves
Spectral method implementation
"""


import numpy as np
from scipy import linalg, sparse, special
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt


save_eigen = True
save_eigen_fname = "./output/eigenmodes_Pm0_cheby1000"
save_pattern_fname = "./output/specspy_TO_Pm0_cheby1000"
# save_pattern_fname = None
# Scipy.sparse.eigs seem to require both matrices nonsingular in generalized eigenproblem
sparse_solver = False


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_TO as cfg


"""Spectral setup"""

# Truncation degree
N_trunc = 1000
# Quadrature
n_quad = N_trunc + 4
xi_quad, wt_quad = special.roots_chebyt(n_quad)
s_quad = (1 + xi_quad)/2
jac = Ls/2

# Precompute values
N_mesh, Xi_mesh = np.meshgrid(np.arange(N_trunc), xi_quad, indexing='ij')

Tn = special.eval_chebyt(N_mesh, Xi_mesh)
dTn_dxi = N_mesh*special.eval_chebyu(N_mesh - 1, Xi_mesh)
dTn_dxi2 = (N_mesh + 1)*special.eval_chebyt(N_mesh, Xi_mesh) - special.eval_chebyu(N_mesh, Xi_mesh)
dTn_dxi2 = N_mesh*dTn_dxi2/(Xi_mesh**2 - 1)

dTn_ds = dTn_dxi/jac
dTn_ds2 = dTn_dxi2/jac/jac


"""Assemble matrices"""

Kuu2 = cfg.k_uu2(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds2[np.newaxis, :, :])
Kuu2 = np.sum(jac*wt_quad*Kuu2, axis=-1)

Kuu1 = cfg.k_uu1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
Kuu1 = np.sum(jac*wt_quad*Kuu1, axis=-1)

Kub1 = cfg.k_ub1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
Kub1 = np.sum(jac*wt_quad*Kub1, axis=-1)

Kub0 = cfg.k_ub0(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
Kub0 = np.sum(jac*wt_quad*Kub0, axis=-1)

Kbu1 = cfg.k_bu1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
Kbu1 = np.sum(jac*wt_quad*Kbu1, axis=-1)

Kbb2 = cfg.k_bb2(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds2[np.newaxis, :, :])
Kbb2 = np.sum(jac*wt_quad*Kbb2, axis=-1)

Kbb1 = cfg.k_bb1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
Kbb1 = np.sum(jac*wt_quad*Kbb1, axis=-1)

Kbb0 = cfg.k_bb0(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
Kbb0 = np.sum(jac*wt_quad*Kbb0, axis=-1)

print("Matrices calculated")

K_mat = np.zeros((2*N_trunc, 2*N_trunc))
K_mat[np.ix_(np.arange(N_trunc), np.arange(N_trunc))] = Kuu2 + Kuu1
K_mat[np.ix_(np.arange(N_trunc), N_trunc + np.arange(N_trunc))] = Kub1 + Kub0
K_mat[np.ix_(N_trunc + np.arange(N_trunc), np.arange(N_trunc))] = Kbu1
K_mat[np.ix_(N_trunc + np.arange(N_trunc), N_trunc + np.arange(N_trunc))] = Kbb2 + Kbb1 + Kbb0

Mu = cfg.m_u(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
Mu = np.sum(jac*wt_quad*Mu, axis=-1)
Mb = cfg.m_b(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
Mb = np.sum(jac*wt_quad*Mb, axis=-1)

M_mat = np.zeros((2*N_trunc, 2*N_trunc))
M_mat[np.ix_(np.arange(N_trunc), np.arange(N_trunc))] = Mu
M_mat[np.ix_(N_trunc + np.arange(N_trunc), N_trunc + np.arange(N_trunc))] = Mb

print("Matrices assembled")


"""Apply Boundary condition"""

K_mat[[N_trunc-2, N_trunc-1, 2*N_trunc-2, 2*N_trunc-1], :] = 0

# Homogeneous Neumann BC for u
K_mat[N_trunc-2, :N_trunc] = np.arange(N_trunc)**2/jac
K_mat[N_trunc-1, :N_trunc] = np.arange(N_trunc)**2/jac
K_mat[N_trunc-1, :N_trunc:2] *= -1
# Homogeneous Dirichlet BC for b
K_mat[2*N_trunc-2, N_trunc:] = 1
K_mat[2*N_trunc-1, N_trunc:] = 1
K_mat[2*N_trunc-1, N_trunc::2] *= -1

M_mat[[N_trunc-2, N_trunc-1, 2*N_trunc-2, 2*N_trunc-1], :] = 0


"""Solve generalized eigenvalue problem"""

if sparse_solver:
    w, v = splinalg.eigs(sparse.csc_array(K_mat), k=10, M=sparse.csc_array(M_mat), which="LR", return_eigenvectors=True, tol=1e-3, maxiter=1000)
else:
    w, v = linalg.eig(K_mat, M_mat)

print("Eigensystem solved")


"""Output"""

import h5py

if save_eigen:
    with h5py.File(save_eigen_fname + ".h5", 'x') as f_write:
        idx_sort = np.argsort(w)
        f_write.create_dataset("degrees", data=np.arange(N_trunc))
        f_write.create_dataset("eigenvals", data=w[idx_sort])
        f_write.create_dataset("eigenfuns", data=v[:, idx_sort])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax = axes[0]
ax.spy(K_mat, precision=1e-12)
# cm = ax.matshow(K_mat)
# plt.colorbar(cm, ax=ax)
ax = axes[1]
ax.spy(M_mat, precision=1e-12)
# cm = ax.matshow(M_mat)
# plt.colorbar(cm, ax=ax)
if save_pattern_fname is not None:
    plt.savefig(save_pattern_fname + '.png', format='png', dpi=150, bbox_inches="tight")
plt.show()

