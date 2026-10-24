CoolSiC-style trench JFET: educational OFF-state 2D Poisson/Green-operator model

Run: pip install numpy scipy matplotlib
     python coolsic_jfet_green_width.py

OUTPUTS
- coolsic_jfet_green_2d_and_vbd.png: device, averaged field, potential, width sweep
- coolsic_jfet_efield_width_comparison.png: green field at +800 V for 0.2, 0.5, 0.8 um P+ widths
- coolsic_jfet_green_efield.png: standalone green field
- coolsic_jfet_potential.png: standalone potential
- coolsic_jfet_width_sweep.csv: voltage criterion vs P+ lateral collar width
- coolsic_jfet_green_data.npz: fields Ex, Ez, raw |E|, 0.20-um averaged |E|, potential, masks and sweep

MODEL
- -nabla^2 V = q*N_D/(epsilon0*epsilon_r) on the free SiC nodes.
- A^(-1) is the discrete Green operator: V = V_fixed + G*(charge + boundary RHS).
- Source top at 0 V, gate metal and p+ collars at -15 V, drain bottom at +V_D.
- Outer lateral boundaries and uncontacted top gaps have zero normal E.
- Assumed εr = 9.7, N_D = 5e15 cm^-3, device 10 um across and 13 um deep.
- Gate trenches at x=+/-2.55 um, half-width 0.68 um, bottom z=2.45 um.
- P+ collar lateral width varied 0.20-0.80 um; P+ bottom thickness fixed at 0.45 um.
- P+ is approximated as equipotential and highly doped; space charge on the p side is neglected.
- The n-type region is assumed depleted wherever the Poisson RHS is applied.
- Metal, dielectric, oxide, 3D geometry, surface traps, transport, gate leakage, incomplete ionization and built-in PN voltage are not modeled.

BREAKDOWN VOLTAGE DISCLAIMER
An actual avalanche breakdown voltage cannot be obtained from Poisson alone.  We report
an illustrative field-threshold voltage Vcrit such that the MAX 0.20-um-averaged
|E| in the free SiC reaches a *selected* reference field 2.5 MV/cm.  Averaging
reduces numerical grid-staircasing artifacts at idealized gate corners.  The
choice of averaging radius and Ecrit changes the numeric result.  The raw field
is also saved.  For device specifications use carrier transport + ionization
integrals, mesh and boundary sensitivity studies and measured calibration.

Numbers with grid 201x261, source 0 V and gate -15 V:
P+ width (um)     Vcrit (V)    channel-gap (um)   1/gap proxy relative
0.20              1275.6       3.34               1.00
0.35              1335.6       3.04               1.10
0.50              1410.5       2.74               1.22
0.65              1483.9       2.44               1.37
0.80              1555.2       2.14               1.56

Mesh check at w=0.5 um: 121x157 -> 1438 V; 161x209 -> 1427 V;
201x261 -> 1410 V; 241x313 -> 1404 V, for this averaging convention.
These results do not imply a particular commercial CoolSiC part's voltage rating.
