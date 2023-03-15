# -*- coding: utf-8 -*-
"""
FEM verification plotting for 2D diffusion equation
Introduction to FEM in Geoscience
Jingtao Min @ ETH Zurich, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

in_fname = "./output/Stokes.npz"
out_fname = None

info = np.load(in_fname)
L2_err, n_elem_trials = info["l2_err"], info["n_elem_dim"]

L2_err = np.sqrt(L2_err)
spacing = 10/n_elem_trials
labels = ["vx", "vy", "p"]

fig = plt.figure(figsize=(18, 6))
for iplot in range(L2_err.shape[1]):
    L2 = L2_err[:, iplot]
    regressor = linear_model.LinearRegression(fit_intercept=True).fit(np.log10(spacing).reshape(-1, 1), np.log10(L2))
    R2 = regressor.score(np.log10(spacing).reshape(-1, 1), np.log10(L2))
    err_est = np.power(10, regressor.predict(np.log10(spacing).reshape(-1, 1)))
    
    plt.subplot(1, 3, iplot+1)
    plt.loglog(spacing, err_est, 'r-', label="FEM solutions")
    plt.loglog(spacing, L2, 'bs', label="Linear fit")

    plt.title("{} - Order of accuracy {:.3f}, $R^2$ = {:.3f}".format(labels[iplot], regressor.coef_[0], R2), fontsize=14)
    
    plt.xlabel("Node spacing", fontsize=14)
    if iplot == 0:
        plt.ylabel("Integrated error ($E_2$)", fontsize=14)
    plt.grid(which="both")
    plt.legend(fontsize=14)

if out_fname is not None:
    plt.savefig(out_fname + ".pdf", format="pdf", bbox_inches="tight")

plt.show()
