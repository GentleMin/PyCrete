"""
Plotting the 1D eigen function solutions
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path


def read_eig(in_name):
    with h5py.File(in_name, 'r') as f:
        xcoord = f["nodes"][()]
        eigvals = f["eigenvals"][()]
        eigfuns = f["eigenfuns"][()]
    return xcoord, eigvals, eigfuns


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


def plot_eigfuns(xcoord, eigvals, eigfuns, out_name, k=10):
    assert eigvals.size == eigfuns.shape[1]
    if k is None:
        k = eigvals.size
    k = k if k <= eigvals.size else eigvals.size
    if isinstance(out_name, list):
        assert len(out_name) == eigvals.size
        out_name_list = out_name
    else:
        out_name_list = [out_name + "{:02d}.png".format(i) for i in range(k)]
    
    prev_eig = 0. + 1j*0.
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    i_plt = 0
    for idx in range(eigvals.size):
        
        if i_plt >= k:
            break
        if np.allclose(eigvals[idx], prev_eig) or np.allclose(np.conj(eigvals[idx]), prev_eig):
            continue
        
        ax = axes[0]
        ax.clear()
        ax.plot(xcoord, np.imag(eigfuns[0::2, idx]), 'r-', label="Im")
        ax.plot(xcoord, np.real(eigfuns[0::2, idx]), 'k-', label="Re")
        ax.legend()
        ax.set_xlabel("s")
        ax.set_title("u")

        ax = axes[1]
        ax.clear()
        ax.plot(xcoord, np.imag(eigfuns[1::2, idx]), 'r-', label="Im")
        ax.plot(xcoord, np.real(eigfuns[1::2, idx]), 'b-', label="Re")
        ax.legend()
        ax.set_xlabel("s")
        ax.set_title("b")
        
        fig.suptitle(r"$\tilde{\omega}=$" + "{:.4f}".format(eigvals[idx]))
        plt.savefig(out_name_list[i_plt], format='png', dpi=128)
        
        prev_eig = eigvals[idx]
        i_plt += 1
    
    plt.close(fig)
    

if __name__ == "__main__":
    in_name = "./output/eigenmodes_n50_v2.h5"
    xcoord, eigvals, eigfuns = read_eig(in_name)
    eigvals, eigfuns = filter_sort_eig(eigvals, eigfuns)
    out_dir = "./output/eigenmodes_n50_v2/"
    Path(out_dir).mkdir(parents=True, exist_ok=False)
    plot_eigfuns(xcoord, eigvals, eigfuns, out_dir + "eigenfunc", k=50)
