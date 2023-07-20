"""
Eikonal equation using scikit-FMM
"""

import skfmm
import numpy as np
import matplotlib.pyplot as plt

"""Set up boundary"""

nx, ny, nz = 101, 101, 101
X, Y, Z = np.meshgrid(np.linspace(0, 20, nx), np.linspace(0, 20, ny), np.linspace(0, 20, nz))
dx = 20/(nx - 1)
Phi = -1*np.ones_like(X)

# Demo boundary (2D only)
# Phi[X > 5] = 1
# Phi[(np.abs(Y - 10) < 2.5) & (X > 2.5)] = 1

# Point source at domain boundary
# Phi[np.isclose(X, 10.) & np.isclose(Y, 1.)] = 1

# Point source at (10, 10, 1)
Phi[np.isclose(X, 10.) & np.isclose(Y, 10.) & np.isclose(Z, 1.)] = 1

"""Set up velocity model"""

Vel = 5*np.ones_like(X)

# Demo velocity (2D only)
# Vel[Y > 10] = 6

# Velocity gradient
V0, V1 = 3., 7.
Vel = (Z - 0)/(20 - 0)*(V1 - V0) + V0
vel_clim = [1., 9.]

# Block velocity anomaly
# Vel[(np.abs(X - 10) <= 4) & (np.abs(Y - 10) <= 4) & (np.abs(Z - 10) <= 4)] = 7.
# vel_clim = [4., 6.]

# Checkerboard example
# X_cycle = np.remainder(X - 0, 6)
# Y_cycle = np.remainder(Y - 1, 6)
# Z_cycle = np.remainder(Z - 5, 6)
# Vel[(np.abs(X_cycle - 1.5) <= 1.5) & (np.abs(Y_cycle - 1.5) <= 1.5) & (np.abs(Z_cycle - 1.5) <= 1.5)] = 6.
# Vel[(np.abs(X_cycle - 4.5) <= 1.5) & (np.abs(Y_cycle - 4.5) <= 1.5) & (np.abs(Z_cycle - 4.5) <= 1.5)] = 6.
# Vel[(np.abs(X_cycle - 4.5) < 1.5) & (np.abs(Y_cycle - 4.5) < 1.5) & (np.abs(Z_cycle - 1.5) < 1.5)] = 4.
# Vel[(np.abs(X_cycle - 4.5) < 1.5) & (np.abs(Y_cycle - 1.5) < 1.5) & (np.abs(Z_cycle - 4.5) < 1.5)] = 4.
# vel_clim = [4., 6.]


"""Calculate (distance and) traveltime"""

d = skfmm.distance(Phi, dx=dx)
traveltime = skfmm.travel_time(Phi, Vel, dx=dx)


"""Visualization"""

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5))

# ax = axes[0]
# ax.contour(X, Y, Phi, [0,], linewidths=(3), colors="black")
# ax.set_title("Boundary location")

# 2D plot

# ax = axes[0]
# ax.contour(X, Y, Phi, [0,], linewidths=(3), colors="black")
# cm = ax.pcolormesh(X, Y, Vel)
# ax.set_title("Velocity model")
# plt.colorbar(cm, ax=ax)

# ax = axes[1]
# ax.contour(X, Y, Phi, [0,], linewidths=(3), colors="black")
# cm = ax.pcolormesh(X, Y, traveltime)
# ax.contour(X, Y, traveltime, np.arange(8, step=0.2), colors="white")
# ax.set_title("Travel time from boundary")
# plt.colorbar(cm, ax=ax)

# 2D slices in 3D domain

y_id = np.arange(ny)[np.isclose(Y[:, 0, 0], 10.)][0]
z_id = np.arange(nz)[np.isclose(Z[0, 0, :], 10.)][0]

ax = axes[0]
ax.contour(X[y_id, :, :], Z[y_id, :, :], traveltime[y_id, :, :], 18, colors="white")
cm = ax.pcolormesh(X[y_id, :, :], Z[y_id, :, :], Vel[y_id, :, :], clim=vel_clim)
ax.set_title("Slice parallel to Oxz")
ax.invert_yaxis()
plt.colorbar(cm, ax=ax)

ax = axes[1]
ax.contour(X[:, :, z_id], Y[:, :, z_id], traveltime[:, :, z_id], 6, colors="white")
cm = ax.pcolormesh(X[:, :, z_id], Y[:, :, z_id], Vel[:, :, z_id], clim=vel_clim)
ax.set_title("Slice parallel to Oxy")
ax.invert_yaxis()
plt.colorbar(cm, ax=ax)

plt.show()
