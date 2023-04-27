"""
1D model for calculating eigenmodes of torsional Alfven waves
Spectral method implementation
"""


import numpy as np
import time, os, yaml
from scipy import linalg, sparse, special
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt


output = True
# output_dir = "./output/eigenmodes_1D/eigenmodes_Pm0_Lu2000_freeslip/"
autogen_output_dir = True

# save_pattern_fname = "./output/specspy_TO_noslip_cheby500"
save_pattern_fname = None

# Scipy.sparse.eigs seem to require both matrices nonsingular in generalized eigenproblem
sparse_solver = False

# whether to use memory-efficient, but not fully scipy-vectorized version
save_memory = False


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_TO as cfg

if autogen_output_dir:
    output_dir = os.path.join("./output/eigenmodes_1D", "Pm{:.0e}__Lu{:.0e}__{}".format(cfg.Pm, cfg.Lu, cfg.bc_u))


"""Spectral setup"""

# Truncation degree
N_trunc = 500
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

start_time = time.time()


"""Assemble matrices"""

if save_memory:
    # Memory-efficient (?) sequential calculation

    Kuu = np.zeros((N_trunc, N_trunc))
    Kub = np.zeros((N_trunc, N_trunc))
    Kbu = np.zeros((N_trunc, N_trunc))
    Kbb = np.zeros((N_trunc, N_trunc))
    Mu = np.zeros((N_trunc, N_trunc))
    Mb = np.zeros((N_trunc, N_trunc))

    for i in range(N_trunc):
        for j in range(N_trunc):
            
            # kuu_quad = Tn[i, :]*(cfg.k_uu2(s_quad)*dTn_ds2[j, :] + cfg.k_uu1(s_quad)*dTn_ds[j, :])
            # Kuu[i, j] = np.sum(jac*wt_quad*kuu_quad)
            # kub_quad = Tn[i, :]*(cfg.k_ub1(s_quad)*dTn_ds[j, :] + cfg.k_ub0(s_quad)*Tn[j, :])
            # Kub[i, j] = np.sum(jac*wt_quad*kub_quad)
            # kbu_quad = Tn[i, :]*(cfg.k_bu1(s_quad)*dTn_ds[j, :])
            # Kbu[i, j] = np.sum(jac*wt_quad*kbu_quad)
            # kbb_quad = Tn[i, :]*(cfg.k_bb2(s_quad)*dTn_ds2[j, :] + cfg.k_bb1(s_quad)*dTn_ds[j, :] + cfg.k_bb0(s_quad)*Tn[j, :])
            # Kbb[i, j] = np.sum(jac*wt_quad*kbb_quad)
            
            # Mu[i, j] = np.sum(jac*wt_quad*Tn[i, :]*cfg.m_u(s_quad)*Tn[j, :])
            # Mb[i, j] = np.sum(jac*wt_quad*Tn[i, :]*cfg.m_b(s_quad)*Tn[j, :])
            
            
            Kuu[i, j] = np.sum(jac*wt_quad*cfg.k_uu2(s_quad)*(Tn[i, :]*dTn_ds2[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_uu1(s_quad)*(Tn[i, :]*dTn_ds[j, :]))
            Kub[i, j] = np.sum(jac*wt_quad*cfg.k_ub1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_ub0(s_quad)*(Tn[i, :]*Tn[j, :]))
            Kbu[i, j] = np.sum(jac*wt_quad*cfg.k_bu1(s_quad)*(Tn[i, :]*dTn_ds[j, :]))
            Kbb[i, j] = np.sum(jac*wt_quad*cfg.k_bb2(s_quad)*(Tn[i, :]*dTn_ds2[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_bb1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_bb0(s_quad)*(Tn[i, :]*Tn[j, :]))
            
            Mu[i, j] = np.sum(jac*wt_quad*cfg.m_u(s_quad)*(Tn[i, :]*Tn[j, :]))
            Mb[i, j] = np.sum(jac*wt_quad*cfg.m_b(s_quad)*(Tn[i, :]*Tn[j, :]))
    
else:
    # Kuu2 = cfg.k_uu2(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds2[np.newaxis, :, :])
    # Kuu1 = cfg.k_uu1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
    
    # Kuu = np.sum(jac*wt_quad*Kuu2, axis=-1) + np.sum(jac*wt_quad*Kuu1, axis=-1)

    # Kub1 = cfg.k_ub1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
    # Kub0 = cfg.k_ub0(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
    
    # Kub = np.sum(jac*wt_quad*Kub1, axis=-1) + np.sum(jac*wt_quad*Kub0, axis=-1)

    # Kbu = cfg.k_bu1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
    # Kbu = np.sum(jac*wt_quad*Kbu, axis=-1)

    # Kbb2 = cfg.k_bb2(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds2[np.newaxis, :, :])
    # Kbb1 = cfg.k_bb1(s_quad)*(Tn[:, np.newaxis, :]*dTn_ds[np.newaxis, :, :])
    # Kbb0 = cfg.k_bb0(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
    
    # Kbb = np.sum(jac*wt_quad*Kbb2, axis=-1) + np.sum(jac*wt_quad*Kbb1, axis=-1) + np.sum(jac*wt_quad*Kbb0, axis=-1)

    # Mu = cfg.m_u(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
    # Mu = np.sum(jac*wt_quad*Mu, axis=-1)
    # Mb = cfg.m_b(s_quad)*(Tn[:, np.newaxis, :]*Tn[np.newaxis, :, :])
    # Mb = np.sum(jac*wt_quad*Mb, axis=-1)
    
    Kuu = cfg.k_uu2(s_quad)*dTn_ds2 + cfg.k_uu1(s_quad)*dTn_ds
    Kuu = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kuu[np.newaxis, :, :], axis=-1)
    print("Kuu assembled")

    Kub = cfg.k_ub1(s_quad)*dTn_ds + cfg.k_ub0(s_quad)*Tn
    Kub = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kub[np.newaxis, :, :], axis=-1)
    print("Kub assembled")

    Kbu = cfg.k_bu1(s_quad)*dTn_ds
    Kbu = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kbu[np.newaxis, :, :], axis=-1)
    print("Kbu assembled")
    
    Kbb = cfg.k_bb2(s_quad)*dTn_ds2 + cfg.k_bb1(s_quad)*dTn_ds + cfg.k_bb0(s_quad)*Tn
    Kbb = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kbb[np.newaxis, :, :], axis=-1)
    print("Kbb assembled")
    
    Mu = np.sum((jac*wt_quad*cfg.m_u(s_quad)*Tn[:, np.newaxis, :])*Tn[np.newaxis, :, :], axis=-1)
    Mb = np.sum((jac*wt_quad*cfg.m_b(s_quad)*Tn[:, np.newaxis, :])*Tn[np.newaxis, :, :], axis=-1)

print("Matrices calculated")

K_mat = np.block([[Kuu, Kub], [Kbu, Kbb]])
M_mat = np.block([[Mu, np.zeros((N_trunc, N_trunc))], [np.zeros((N_trunc, N_trunc)), Mb]])

print("Matrices assembled")


"""Apply Boundary condition"""

K_mat[[N_trunc-2, N_trunc-1, 2*N_trunc-2, 2*N_trunc-1], :] = 0

if cfg.bc_u == "noslip":
    # Homogeneous Dirichlet BC at 1 for u
    K_mat[N_trunc-2, :N_trunc] = np.arange(N_trunc)**2/jac
    K_mat[N_trunc-2, :N_trunc:2] *= -1
    K_mat[N_trunc-1, :N_trunc] = 1
elif cfg.bc_u == "freeslip":
    # Homogeneous Neumann BC for u
    K_mat[N_trunc-2, :N_trunc] = np.arange(N_trunc)**2/jac
    K_mat[N_trunc-1, :N_trunc] = np.arange(N_trunc)**2/jac
    K_mat[N_trunc-1, :N_trunc:2] *= -1

if cfg.bc_b == "insulating":
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

end_time = time.time()
print("Time = {:.2f} s".format(end_time - start_time))


"""Output"""

import h5py

# np.save("./output/K_mat_vectorized_v3.npy", arr=K_mat)
# np.save("./output/M_mat_vectorized_v3.npy", arr=M_mat)

if output:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_fname = "cheby{:d}_redvec.h5".format(N_trunc)
    with h5py.File(os.path.join(output_dir, save_fname), 'x') as f_write:
        idx_sort = np.argsort(w)
        f_write.create_dataset("degrees", data=np.arange(N_trunc))
        f_write.create_dataset("eigenvals", data=w[idx_sort])
        f_write.create_dataset("eigenfuns", data=v[:, idx_sort])
    cfg_fname = "cheby{:d}_redvec.yaml".format(N_trunc)
    with open(os.path.join(output_dir, cfg_fname), 'x') as f_cfg:
        yaml.dump(data={"Lu": cfg.Lu, 
                        "Pm": cfg.Pm, 
                        "Truncation degree": N_trunc, 
                        "Magnetic BC": cfg.bc_b, 
                        "Velocity BC": cfg.bc_u, 
                        "Method": "Classical Chebyshev, classical tau", 
                        "Sparse solver": sparse_solver}, 
                  stream=f_cfg, default_flow_style=False, sort_keys=False)
    print("Eigensolutions saved to " + output_dir)

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

