"""
Plotting the 1D eigen function solutions
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from spectral import basis


def read_eig_fem(in_name):
    with h5py.File(in_name, 'r') as f:
        xcoord = f["nodes"][()]
        eigvals = f["eigenvals"][()]
        eigfuns = f["eigenfuns"][()]
    return xcoord, eigvals, eigfuns

def read_eig_spectrum(in_name):
    with h5py.File(in_name, 'r') as f:
        degrees = f["degrees"][()]
        eigvals = f["eigenvals"][()]
        eigfuns = f["eigenfuns"][()]
    return degrees, eigvals, eigfuns

def assemble_spectral_funcs(degrees, xcoord, coeffs):
    cheby_eval = basis.ChebyshevTSpace(degrees, xcoord)
    eigenmodes = np.zeros((2*xcoord.size, coeffs.shape[1]), dtype=np.complex128)
    eigenmodes[::2, :] = np.array([cheby_eval(coeffs[:degrees.size, i]) for i in range(coeffs.shape[1])]).T
    eigenmodes[1::2, :] = np.array([cheby_eval(coeffs[degrees.size:, i]) for i in range(coeffs.shape[1])]).T
    return eigenmodes

def filter_sort_eig(eigvals, eigfuns):
    # Filter the slow-decaying modes (these are usually the non-spurious ones)
    idx_filtered = np.abs(np.imag(eigvals)) > 0.5*np.abs(np.real(eigvals))
    eigvals = eigvals[idx_filtered]
    eigfuns = eigfuns[:, idx_filtered]
    # Sort by real part, and then by imaginary part
    idx_sorted = sorted(list(range(eigvals.size)), key=lambda i: (-np.real(eigvals[i]), np.abs(np.imag(eigvals[i]))))
    eigvals = eigvals[idx_sorted]
    eigfuns = eigfuns[:, idx_sorted]
    return eigvals, eigfuns

def plot_solution(xcoord, eigfun, yvar=None, ax=None):
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

def plot_batch_eigmodes(xcoord, eigvals, eigfuns, out_name, k=10, normalized=False):
    assert eigvals.size == eigfuns.shape[1]
    if k is None:
        k = eigvals.size
    k = k if k <= eigvals.size else eigvals.size
    if isinstance(out_name, list):
        assert len(out_name) == eigvals.size
        out_name_list = out_name
    else:
        out_name_list = [out_name + "{:02d}.png".format(i) for i in range(k)]
    
    if normalized:
        eigfuns = eigfuns/np.max(eigfuns, axis=0)
    
    prev_eig = 0. + 1j*0.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    i_plt = 0
    for idx in range(eigvals.size):
        
        if i_plt >= k:
            break
        if np.allclose(eigvals[idx], prev_eig) or np.allclose(np.conj(eigvals[idx]), prev_eig):
            continue
        
        plot_solution(xcoord, eigfuns[0::2, idx], yvar='u', ax=axes[0])
        plot_solution(xcoord, eigfuns[1::2, idx], yvar='b', ax=axes[1])        
        fig.suptitle(r"$\tilde{\omega}=$" + "{:.4f}".format(eigvals[idx]))
        plt.savefig(out_name_list[i_plt], format='png', dpi=128)
        
        prev_eig = eigvals[idx]
        i_plt += 1
    
    plt.close(fig)

def multimode_stepping(xcoord, eigvals, eigfuns, weights, t_max, out_name, dt=0.1):
    assert eigvals.size == eigfuns.shape[1]
    normalizer = np.max(eigfuns, axis=0)
    eigfuns = eigfuns/normalizer
    t_array = np.arange(0, t_max, dt)
    out_name_list = [out_name + "{:02d}.png".format(i) for i in range(t_array.size)]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    for i_plt, t in enumerate(t_array):
        
        mixed_model = np.sum(weights*np.exp(eigvals*t)*eigfuns, axis=1)
        plot_solution(xcoord, mixed_model[0::2], yvar='u', ax=axes[0])
        axes[0].set_ylim([-15, 15])
        plot_solution(xcoord, mixed_model[1::2], yvar='b', ax=axes[1])
        axes[1].set_ylim([-0.02, 0.02])
        fig.suptitle("t={:.2f}".format(t))
        plt.savefig(out_name_list[i_plt], format="png", dpi=128)


def routine_plot_eigenmodes_fem(in_name, out_dir):
    
    xcoord, eigvals, eigfuns = read_eig_fem(in_name)
    eigvals, eigfuns = filter_sort_eig(eigvals, eigfuns)    
    Path(out_dir).mkdir(parents=True, exist_ok=False)
    plot_batch_eigmodes(xcoord, eigvals, eigfuns, out_dir + "eigenfunc", k=50)

def routine_multimode_stepping_fem(in_name, out_dir):
    
    xcoord, eigvals, eigfuns = read_eig_fem(in_name)
    eigvals, eigfuns = filter_sort_eig(eigvals, eigfuns)
    Path(out_dir).mkdir(parents=True, exist_ok=False)    
    multimode_stepping(xcoord, eigvals[:20], eigfuns[:, :20], weights=np.ones(20), t_max=5, 
                       out_name=out_dir + "snap", dt=0.1)
    
def routine_plot_eigenmodes_spectral(in_name, out_dir):
    
    degrees, eigvals, eigfuns = read_eig_spectrum(in_name)
    xi_array = np.linspace(-1, 1, num=100)
    s_array = (1 + xi_array)/2
    eigvals, eigfuns = filter_sort_eig(eigvals, eigfuns)    
    eigenmodes = assemble_spectral_funcs(degrees, xi_array, eigfuns)
    Path(out_dir).mkdir(parents=True, exist_ok=False)
    plot_batch_eigmodes(s_array, eigvals, eigenmodes, out_dir + "eigenfunc", k=50)


if __name__ == "__main__":
    
    routine_plot_eigenmodes_spectral(in_name="./output/eigenmodes_Pm0_cheby50.h5", 
                                     out_dir="./output/eigenmodes_Pm0_cheby50/")

    