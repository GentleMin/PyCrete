"""
1D heat equation for coupled 2-phase charging
Chebyshev spectral method
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

# whether to compute the steady-state solution
steady_state = False


"""Physics setup"""

s_range = np.array([0, 1])
Ls = s_range[1] - s_range[0]
import config_2phase as cfg

if autogen_output_dir:
    output_dir = os.path.join("./output/thermal_2phase_1D", "T_steady_flux")


"""Time-stepping setup"""
dt = 1e-3
tmax = 1.
step_verbose = 50

"""Spectral setup"""

# Truncation degree
N_trunc = 100
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
            
            Kuu[i, j] = np.sum(jac*wt_quad*cfg.k_uu2(s_quad)*(Tn[i, :]*dTn_ds2[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_uu1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_uu0(s_quad)*(Tn[i, :]*Tn[j, :]))
            Kub[i, j] = np.sum(jac*wt_quad*cfg.k_ub1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_ub0(s_quad)*(Tn[i, :]*Tn[j, :]))
            Kbu[i, j] = np.sum(jac*wt_quad*cfg.k_bu1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_bu0(s_quad)*(Tn[i, :]*Tn[j, :]))
            Kbb[i, j] = np.sum(jac*wt_quad*cfg.k_bb2(s_quad)*(Tn[i, :]*dTn_ds2[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_bb1(s_quad)*(Tn[i, :]*dTn_ds[j, :])) + \
                np.sum(jac*wt_quad*cfg.k_bb0(s_quad)*(Tn[i, :]*Tn[j, :]))
            
            Mu[i, j] = np.sum(jac*wt_quad*cfg.m_u(s_quad)*(Tn[i, :]*Tn[j, :]))
            Mb[i, j] = np.sum(jac*wt_quad*cfg.m_b(s_quad)*(Tn[i, :]*Tn[j, :]))
    
else:
    
    Kuu = cfg.k_uu2(s_quad)*dTn_ds2 + cfg.k_uu1(s_quad)*dTn_ds + cfg.k_uu0(s_quad)*Tn
    Kuu = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kuu[np.newaxis, :, :], axis=-1)
    print("Kuu assembled")

    Kub = cfg.k_ub1(s_quad)*dTn_ds + cfg.k_ub0(s_quad)*Tn
    Kub = np.sum((jac*wt_quad*Tn[:, np.newaxis, :])*Kub[np.newaxis, :, :], axis=-1)
    print("Kub assembled")

    Kbu = cfg.k_bu1(s_quad)*dTn_ds + cfg.k_bu0(s_quad)*Tn
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

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax = axes[0]
# ax.spy(K_mat, precision=1e-12)
cm = ax.matshow(K_mat)
plt.colorbar(cm, ax=ax)
ax = axes[1]
# ax.spy(M_mat, precision=1e-12)
cm = ax.matshow(M_mat)
plt.colorbar(cm, ax=ax)
if save_pattern_fname is not None:
    plt.savefig(save_pattern_fname + '.png', format='png', dpi=150, bbox_inches="tight")
plt.show()

# exit()


"""Time stepping or steady-state solution"""

if steady_state:
    
    L_mat = K_mat
    f_vec = np.zeros(2*N_trunc)

    L_mat[N_trunc-2, :N_trunc] = 1
    L_mat[N_trunc-2, 1:N_trunc:2] *= -1
    f_vec[N_trunc-2] = 1.

    L_mat[N_trunc-1, :N_trunc] = np.arange(N_trunc)**2/jac
    f_vec[N_trunc-1] = -1.

    L_mat[2*N_trunc-2, N_trunc:] = 1
    L_mat[2*N_trunc-2, N_trunc+1::2] *= -1
    f_vec[2*N_trunc-2] = 0.

    L_mat[2*N_trunc-1, N_trunc:] = np.arange(N_trunc)**2/jac
    f_vec[2*N_trunc-1] = 1.

    solution_array = np.linalg.solve(L_mat, f_vec)

else:

    t_array = np.arange(0, tmax, dt)
    solution_array = np.zeros((t_array.size, N_trunc*2))

    # Initial condition
    solution_array[0][[0, N_trunc]] = 1.

    for t_id in range(t_array.size - 1):
        L_mat = M_mat - dt*K_mat
        f_vec = M_mat @ solution_array[t_id]
        
        # Boundary conditions
        L_mat[N_trunc-2, :N_trunc] = 1
        L_mat[N_trunc-2, 1:N_trunc:2] *= -1
        f_vec[N_trunc-2] = cfg.bound_left(t_array[t_id+1])
        
        L_mat[N_trunc-1, :N_trunc] = np.arange(N_trunc)**2/jac
        f_vec[N_trunc-1] = 0
        
        L_mat[2*N_trunc-2, N_trunc:] = np.arange(N_trunc)**2/jac
        L_mat[2*N_trunc-2, N_trunc::2] *= -1
        f_vec[2*N_trunc-2] = 0
        
        L_mat[2*N_trunc-1, N_trunc:] = np.arange(N_trunc)**2/jac
        f_vec[2*N_trunc-1] = 0
        
        # print(f_vec)
        
        solution_step = np.linalg.solve(L_mat, f_vec)
        solution_array[t_id+1, :] = solution_step
        
        if (t_id + 1) % step_verbose == 0:
            print("{:d}/{:d} steps solved.".format(t_id + 1, t_array.size))

end_time = time.time()
print("Time = {:.2f} s".format(end_time - start_time))


"""Output"""

import h5py

# np.save("./output/K_mat_vectorized_v3.npy", arr=K_mat)
# np.save("./output/M_mat_vectorized_v3.npy", arr=M_mat)

if output:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_fname = "solution_cheby{:d}.h5".format(N_trunc)
    with h5py.File(os.path.join(output_dir, save_fname), 'x') as f_write:
        f_write.create_dataset("solution", data=solution_array)
    print("Solutions saved to " + output_dir)

