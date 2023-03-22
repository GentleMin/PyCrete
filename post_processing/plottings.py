"""Plotting utilities
"""


import numpy as np
import matplotlib.pyplot as plt


def plot_function_1D(xcoord, eigfun, yvar=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(nrows=1, ncols=1)
    ax.clear()
    ax.plot(xcoord, np.imag(eigfun), 'r-', label="Im")
    ax.plot(xcoord, np.real(eigfun), 'b-', label="Re")
    ax.legend()
    ax.set_xlabel('s')
    if yvar is not None:
        ax.set_title(yvar)
    return ax

