# -*- coding: utf-8 -*-
"""
FEM verification plotting for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

in_fname_list = ["./output/Diffuse_dyn_biquad_5.npz", 
            "./output/Diffuse_dyn_biquad_4e-01.npz", 
            "./output/Diffuse_dyn_biquad_2e-01.npz", 
            "./output/Diffuse_dyn_biquad_1e-01.npz"]
labels = ["dt = 0.5", "dt = 0.4", "dt = 0.2", "dt = 0.1"]
colors = ["red", "mediumvioletred", "blueviolet", "blue"]
out_fname = "./output/Diffuse_dyn"

L2_errors = list()

for in_fname in in_fname_list:
    info = np.load(in_fname)
    L2_err, n_elem_trials = info["l2_err"], info["n_elem_dim"]
    L2_errors.append(L2_err)

L2_errors = np.sqrt(np.array(L2_errors))
spacing = 10/n_elem_trials

plt.figure(figsize=(6, 6))

regressor = linear_model.LinearRegression(fit_intercept=True).fit(np.log10(spacing[:3]).reshape(-1, 1), np.log10(L2_errors[-1, :3]))
R2_1 = regressor.score(np.log10(spacing[:3]).reshape(-1, 1), np.log10(L2_errors[-1, :3]))
err_est = np.power(10, regressor.predict(np.log10(spacing[:4]).reshape(-1, 1)))
order_1 = regressor.coef_[0]
plt.scatter(spacing[:3], L2_errors[-1, :3], s=150, edgecolors="red", facecolors="none")
plt.loglog(spacing[:4], err_est, 'r-', label="Fitting")

regressor = linear_model.LinearRegression(fit_intercept=True).fit(np.log10(spacing[2:]).reshape(-1, 1), np.log10(L2_errors[[1, 2, 3], [2, 3, 4]]))
R2_2 = regressor.score(np.log10(spacing[2:]).reshape(-1, 1), np.log10(L2_errors[[1, 2, 3], [2, 3, 4]]))
err_est = np.power(10, regressor.predict(np.log10(spacing[1:]).reshape(-1, 1)))
order_2 = regressor.coef_[0]
plt.scatter(spacing[2:], L2_errors[[1, 2, 3], [2, 3, 4]], s=150, edgecolors="black", facecolors="none")
plt.loglog(spacing[1:], err_est, 'k-', label="Fitting, order={:.3f}".format(order_2))

for i_err in range(len(in_fname_list)):
    plt.loglog(spacing, L2_errors[i_err, :], 's', color=colors[i_err], label=labels[i_err])

plt.xlabel("Node spacing", fontsize=14)
plt.ylabel("Integrated error ($E_2$)", fontsize=14)
plt.grid(which="both")
plt.legend(fontsize=14)
plt.title("Order of accuracy {:.3f}, $R^2$ = {:.3f}".format(order_1, R2_1), fontsize=16)

if out_fname is not None:
    plt.savefig(out_fname + ".pdf", format="pdf", bbox_inches="tight")

plt.show()
