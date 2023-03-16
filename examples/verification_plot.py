# -*- coding: utf-8 -*-
"""
FEM verification plotting for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

in_fname = "./output/L2_array_bilinear_sparse.npz"
# in_fname = "./output/L2_array_biquadratic_sparse.npz"
out_fname = None

info = np.load(in_fname)
L2_err, n_elem_trials = info["l2_err"], info["n_elem_dim"]
print(L2_err)

L2_err = L2_err[1:]
n_elem_trials = n_elem_trials[1:]
L2_err = np.sqrt(L2_err)
spacing = 10/n_elem_trials

regressor = linear_model.LinearRegression(fit_intercept=True).fit(np.log10(spacing).reshape(-1, 1), np.log10(L2_err))
R2 = regressor.score(np.log10(spacing).reshape(-1, 1), np.log10(L2_err))
err_est = np.power(10, regressor.predict(np.log10(spacing).reshape(-1, 1)))

plt.figure(figsize=(6, 6))
plt.loglog(spacing, err_est, 'r-', label="FEM solutions")
plt.loglog(spacing, L2_err, 'bs', label="Linear fit")

plt.title("Order of accuracy {:.3f}, $R^2$ = {:.3f}".format(regressor.coef_[0], R2), fontsize=16)

plt.xlabel("Node spacing", fontsize=14)
plt.ylabel("Integrated error ($E_2$)", fontsize=14)
plt.grid(which="both")
plt.legend(fontsize=14)
if out_fname is not None:
    plt.savefig(out_fname + ".pdf", format="pdf", bbox_inches="tight")

plt.show()
