"""
Eigenvalue problem output postprocessing utilities

JTM @ ETH Zurich, 2023-3-29
"""


import numpy as np
import h5py
import warnings


def filter_sort_eig(eigvals, eigfuns=None, threshold=0.5):
    # Filter the slow-decaying modes (these are usually the non-spurious ones)
    idx_filtered = np.abs(np.imag(eigvals)) > threshold*np.abs(np.real(eigvals))
    eigvals = eigvals[idx_filtered]
    if eigfuns is not None:
        eigfuns = eigfuns[:, idx_filtered]
    # Sort by real part, and then by imaginary part
    idx_sorted = sorted(list(range(eigvals.size)), key=lambda i: (-np.real(eigvals[i]), np.abs(np.imag(eigvals[i]))))
    eigvals = eigvals[idx_sorted]
    if eigfuns is not None:
        eigfuns = eigfuns[:, idx_sorted]
        return eigvals, eigfuns
    else:
        return eigvals


class EigenTracker:
    
    def __init__(self, eigen_seed, r_tol=1e-2, a_tol=0., non_mono_warning=True) -> None:
        self.eigen_seed = eigen_seed
        self.r_tol = r_tol
        self.a_tol = a_tol
        self.warning = non_mono_warning
        self.threshold = self.update_threshold()
        self.prev_diff = np.nan
    
    def update_threshold(self):
        return self.r_tol*np.abs(self.eigen_seed) + self.a_tol
    
    def track_next(self, eigen_array):
        min_idx = np.argmin(np.abs(eigen_array - self.eigen_seed))
        # min_idx_conj = np.argmin(np.abs(np.conj(eigen_array) - self.eigen_seed))
        # min_idx = min_idx_conj if min_idx_conj < min_idx else min_idx
        eigen_candidate = eigen_array[min_idx]
        if np.abs(eigen_candidate - self.eigen_seed) > self.threshold:
            warnings.warn("Large discrepancies")
        if not np.isnan(self.prev_diff) and np.abs(eigen_candidate - self.eigen_seed) > self.prev_diff:
            warnings.warn("Non-monotonic eigenvalue difference - potential non-convergence")
        self.prev_diff = np.abs(self.eigen_seed - eigen_candidate)
        self.eigen_seed = eigen_candidate
        self.update_threshold()
        return min_idx, eigen_candidate


def track_eigenvalues(f_eigen_list, k=20, reverse_track=False):
    assert len(f_eigen_list) >= 1
    if reverse_track:
        f_eigen_list = list(reversed(f_eigen_list))
    with h5py.File(f_eigen_list[0], 'r') as f:
        eigen_vals = f["eigenvals"][()]
        eigen_funs = f["eigenfuns"][()]
    eigen_vals, _ = filter_sort_eig(eigen_vals, eigen_funs)
    eigen_vals = eigen_vals[:k]
    trackers = [EigenTracker(eigenval) for eigenval in eigen_vals]
    tracked_val = np.zeros((k, len(f_eigen_list)), dtype=np.complex128)
    tracked_idx = np.zeros((k, len(f_eigen_list)), dtype=np.int32)
    for i_file in range(len(f_eigen_list)):
        with h5py.File(f_eigen_list[i_file], 'r') as f:
            eigen_vals = f["eigenvals"][()]
        track_temp = [tracker.track_next(eigen_vals) for tracker in trackers]
        tracked_val[:, i_file] = np.array([it[1] for it in track_temp])
        tracked_idx[:, i_file] = np.array([it[0] for it in track_temp], dtype=np.int32)
    if reverse_track:
        tracked_val = np.flip(tracked_val, axis=1)
        tracked_idx = np.flip(tracked_idx, axis=1)
    return tracked_idx, tracked_val
        
        

