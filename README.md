# PyCrete: Discrete Numerical PDE Solvers in Python

Under development...

This repo started out as a repo for *Introduction to FEM in Geoscience* block course @ ETH Zurich, Summer 2022.

Dependencies of the modules:
```
NumPy
SciPy
Matplotlib
Pytorch (only needed for PINNs)
```

Additional dependencies of the examples
```
scikit-learn
scikit-fmm
```

---

## Code arrangement

The following utilities are current added:
- 1-D and 2-D utilities for Finite-Element Method (FEM)
- 1-D utilities for Spectral Method
- Utilities for Physics-informed Neural Networks (PINN)

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
