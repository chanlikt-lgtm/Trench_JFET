#!/usr/bin/env python3
"""Vertical 4H-SiC JFET OFF-state blocking: VGS = -4 V, VDS = 100 V.

Depletion-approximation 2D Poisson (same framework as the CoolSiC bundle),
appropriate for a reverse-biased blocking bias where the nonlinear-Boltzmann
equilibrium solver is out of scope.

- The n- drift and n-channel are assumed DEPLETED: fixed ionized-donor space
  charge rho = +q*ND, no mobile carriers.
- P+ columns are heavily doped and reverse-biased at low leakage, so they are
  treated as EQUIPOTENTIAL at the gate terminal potential VGS (Dirichlet).
- Metal BC: source = 0 V, drain = +VDS, gate metal = VGS.
- Outer lateral boundaries: zero normal field (Neumann).
- Linear Poisson, single sparse solve. NOT an avalanche / transport solution;
  peak |E| at idealized doping corners is mesh-dependent.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

OUT = Path(__file__).resolve().parent
q = 1.602176634e-19
EPS = 8.8541878128e-12 * 9.7
VGS = -4.0
VDS = 100.0
N_CH = 1e16
N_DRIFT = 5e15
N_PLUS = 1e18

x = np.linspace(-3, 3, 121)
z = np.linspace(0, 8, 161)
X, Z = np.meshgrid(x, z)
nz, nx = X.shape
dx = x[1] - x[0]
dz = z[1] - z[0]

pcol = (np.abs(X) >= 0.90 - 1e-8) & (np.abs(X) <= 1.55 + 1e-8) & (Z >= 0.55 - 1e-8) & (Z <= 3.05 + 1e-8)
src_plus = (np.abs(X) < 0.9 + 1e-8) & (Z <= 0.55 + 1e-8)
drn_plus = (Z >= 7.55 - 1e-8)
channel = (np.abs(X) < 0.9 - 1e-8) & (Z >= 0.55) & (Z <= 3.05)
ND = np.full(X.shape, N_DRIFT, dtype=float)
ND[channel] = N_CH
ND[src_plus | drn_plus] = N_PLUS
ND[pcol] = 0.0

# Dirichlet nodes: metal contacts + equipotential P+ columns + neutral n+ blocks.
# The n+ source/drain are heavily doped and quasi-neutral: equipotential
# contacts, NOT depleted space charge.
gate = (np.abs(np.abs(X) - 1.55) <= dx * 0.49) & (Z >= 0.80 - 1e-8) & (Z <= 1.60 + 1e-8)
fix = src_plus | drn_plus | gate | pcol

values = np.zeros(X.shape)
values[src_plus] = 0.0               # n+ source block at source terminal
values[drn_plus] = VDS               # n+ drain buffer at drain terminal
values[gate | pcol] = VGS            # P+ columns tied to gate terminal

# -nabla^2 V = q*ND/eps only on the depleted n- drift / n-channel (ND <= N_CH).
coef_rho = q * 1e-6 / EPS           # multiply by ND[cm^-3] -> V/um^2
free = np.argwhere(~fix)
idx = -np.ones(X.shape, int)
idx[~fix] = np.arange(len(free))
NN = len(free)
A = lil_matrix((NN, NN), dtype=float)
rhs = np.zeros(NN)
neighbors = [(-1, 0, 1 / dz**2), (1, 0, 1 / dz**2), (0, -1, 1 / dx**2), (0, 1, 1 / dx**2)]
for k, (i, j) in enumerate(free):
    diagonal = 0.0
    rhs[k] = coef_rho * ND[i, j]     # ionized donor space charge
    for di, dj, coef in neighbors:
        ii, jj = i + di, j + dj
        if not (0 <= ii < nz and 0 <= jj < nx):
            continue                 # zero-normal-field Neumann
        diagonal += coef
        if fix[ii, jj]:
            rhs[k] += coef * values[ii, jj]
        else:
            A[k, idx[ii, jj]] = -coef
    A[k, k] = diagonal
A = A.tocsc()
lu = splu(A)
sol = lu.solve(rhs)

V = values.copy()
V[~fix] = sol
resid = np.max(np.abs(A @ sol - rhs)) / max(np.max(np.abs(rhs)), 1)

dVdz, dVdx = np.gradient(V, dz, dx, edge_order=2)
EX = -dVdx * 0.01                    # V/um -> MV/cm
EZ = -dVdz * 0.01
EMAG = np.hypot(EX, EZ)
# field only meaningful in the semiconductor, exclude equipotential P+ interior
active = ~pcol
peakE = float(EMAG[active].max())
iz, ix = np.unravel_index(np.argmax(np.where(active, EMAG, 0)), EMAG.shape)

print(f"VGS={VGS} V  VDS={VDS} V   linear residual={resid:.2e}")
print(f"2D peak |E| (excl. P+ interior) = {peakE:.3f} MV/cm at x={x[ix]:.2f}, z={z[iz]:.2f} um")
print(f"V range in free region: [{V[~fix].min():.2f}, {V[~fix].max():.2f}] V")

# ---- figures ----
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12})
fig = plt.figure(figsize=(14, 8.9), facecolor='white')
gs = fig.add_gridspec(1, 3, left=.06, right=.97, top=.855, bottom=.155, wspace=.30)
axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
fig.suptitle('Vertical 4H-SiC JFET  |  OFF-state blocking   VGS = -4 V,  VDS = 100 V',
             fontsize=15, fontweight='bold', y=.98)
fig.text(.5, .928, 'Depletion-approximation 2D Poisson (ionized-donor space charge; P+ columns equipotential at gate)',
         ha='center', fontsize=11, color='#45556c')
for ax, title in zip(axs, ['A  Device / boundary conditions', 'B  2D electric-field magnitude', 'C  2D electrostatic potential']):
    ax.set_title(title, pad=11, fontweight='bold', color='#15344f')
    ax.set_xlim(-3, 3); ax.set_ylim(8.1, -.13)
    ax.set_xlabel('Lateral x (µm)'); ax.set_ylabel('Depth z (µm)')
    ax.set_aspect('equal'); ax.set_xticks([-3, -1.5, 0, 1.5, 3]); ax.tick_params(labelsize=9)

ax = axs[0]
ax.set_facecolor('#f4e6b9')
for m, c in ((src_plus, '#f8c342'), (drn_plus, '#efb23c'), (pcol, '#e59dab')):
    ax.contourf(X, Z, m.astype(float), levels=[.5, 1.5], colors=[c])
for xx in (-1.88, 1.55):
    ax.add_patch(patches.Rectangle((xx, .80), .33, .8, facecolor='#48535f', edgecolor='#263447', lw=.9, zorder=6))
ax.add_patch(patches.Rectangle((-.8, -.10), 1.6, .16, facecolor='#48535f', zorder=6))
ax.add_patch(patches.Rectangle((-3, 7.97), 6, .16, facecolor='#48535f', zorder=6))
ax.text(0, .31, 'n+ source  0 V', ha='center', va='center', fontsize=8.5, weight='bold')
ax.text(0, 1.8, 'n-channel\n(depleted,\npinched off)', ha='center', va='center', fontsize=9, weight='bold', color='#123f6a')
for xx in (-1.22, 1.22):
    ax.text(xx, 2.2, 'P+\n−4 V', ha='center', fontsize=8.5, weight='bold', color='#73313c')
ax.text(0, 5.4, 'n− SiC drift\n(depleted)', ha='center', fontsize=10, weight='bold', color='#53491c')
ax.text(0, 7.75, 'D = +100 V', ha='center', fontsize=9, weight='bold', color='white', zorder=8)

cmap = LinearSegmentedColormap.from_list('greens', ['#053426', '#116343', '#16a75b', '#8add59', '#f0fb66'])
upper = max(.5, np.ceil(peakE * 2) / 2)
im = axs[1].pcolormesh(X, Z, np.ma.array(EMAG, mask=pcol), cmap=cmap, vmin=0, vmax=upper, shading='auto', rasterized=True)
axs[1].contour(X, Z, np.ma.array(EMAG, mask=pcol), levels=np.linspace(upper / 12, upper, 12), colors='white', linewidths=.55, alpha=.8)
axs[1].contour(X, Z, pcol.astype(float), levels=[.5], colors='#ffd4ab', linewidths=1.2)
axs[1].plot(x[ix], z[iz], 'x', ms=10, mew=2, color='#fff2a8', zorder=10)
cb = fig.colorbar(im, ax=axs[1], orientation='horizontal', fraction=.043, pad=.095, aspect=26)
cb.set_label('Magnitude |E| (MV/cm)', fontsize=10)

im2 = axs[2].pcolormesh(X, Z, V, cmap='turbo', vmin=0, vmax=VDS, shading='auto', rasterized=True)
axs[2].contour(X, Z, np.ma.array(V, mask=fix), levels=np.linspace(0, VDS, 16), colors='white', linewidths=.5, alpha=.8)
axs[2].contour(X, Z, pcol.astype(float), levels=[.5], colors='#ffe19a', linewidths=1.1)
cb2 = fig.colorbar(im2, ax=axs[2], orientation='horizontal', fraction=.043, pad=.095, aspect=26)
cb2.set_label('Electrostatic potential (V)', fontsize=10)
fig.text(.5, .055, 'P+ NA(tied to gate) · n channel ND = 1e16 · n- drift ND = 5e15 cm⁻³ · 4H-SiC εr = 9.7  |  depletion approx., NOT avalanche BV',
         ha='center', fontsize=9, color='#486176')
fig.savefig(OUT / 'offstate_vgs-4_vds100_2d.png', dpi=200, facecolor='white')
fig.savefig(OUT / 'offstate_vgs-4_vds100_2d.svg')
plt.close(fig)

np.savez_compressed(OUT / 'offstate_vgs-4_vds100_data.npz', x_um=x, z_um=z, potential_V=V,
                    Ex_MV_cm=EX, Ez_MV_cm=EZ, E_MV_cm=EMAG, ND_cm3=ND, Pplus=pcol,
                    gate=gate, VGS=VGS, VDS=VDS, peakE=peakE)
with (OUT / 'offstate_vgs-4_vds100_results.txt').open('w') as f:
    f.write('Vertical 4H-SiC JFET OFF-state blocking (depletion-approximation 2D Poisson)\n')
    f.write(f'VGS = {VGS} V ; VDS = {VDS} V\n')
    f.write('n- drift and channel depleted (fixed ionized donors); P+ columns equipotential at gate.\n')
    f.write(f'Linear residual = {resid:.2e}\n')
    f.write(f'2D peak |E| (excl. P+ interior) = {peakE:.4f} MV/cm at x={x[ix]:.3f}, z={z[iz]:.3f} um\n')
    f.write('Mesh-dependent at idealized doping corners; NOT a transport/avalanche breakdown result.\n')
print('wrote figure, SVG, NPZ and results.txt ->', OUT)
