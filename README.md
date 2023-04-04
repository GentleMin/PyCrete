# PyCrete: Discrete Numerical PDE Solvers in Python

Under development...

This repo started out as a repo for *Introduction to FEM in Geoscience* block course @ ETH Zurich, Summer 2022.

The following python packages are needed for the environment:
```
NumPy
SciPy
Scikit-Learn
Matplotlib
```

---

## Code arrangement

Currently only utilities for 1-D and 2-D finite-element method are added.

---

## Methodology

### Finite-Element Method

The utilities are collected in module `fem`.

### Spectral Method

The utilities are collected in module `spectral`.

### Physics-Informed Neural Network

The utilities are collected in module `physics_nn`.
For this module in particular, the python package `pytorch` is needed.
The implementation relies heavily on the framework, in particular the auto-differentiation (`autograd`) module.

---

## Examples and model problems
