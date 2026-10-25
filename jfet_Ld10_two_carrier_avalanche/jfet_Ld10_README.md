# JFET with 10 µm N-drift: 2D coupled e/h avalanche sweep

## Geometry and fixed parameters

- Si at 300 K; uniform N drift doping `ND=1e16 cm^-3`.
- Finite top and bottom P+ Si pillars, each length `L=1 um`.
- New N drift length `Ld=10 um`, no gate along drift region.
- Total device height fixed `H=2 um`, so central gap `h=H-2*a`.
- P+ thickness sweep `a=0.2 ... 0.8 um`, 0.1 um steps.
- P+ doping sweep `NA=3.162e16 ... 3.162e18 cm^-3` (eight points).
- N+ source/drain doping `1e18 cm^-3`; lateral N+ widths `ws=wd=0.2 um` are *assumed*, not user-specified measurements.
- Ideal ohmic source `0 V`, drain swept from `0 V`, top/bottom P+ gate `-4 V`.
- Silicon contact electrostatic offsets derived from nondegenerate Boltzmann statistics.
- Exterior non-contact surfaces have zero normal flux. P+/N junctions are internal interfaces with charge and potential continuity.

## Equations and numerical scheme

The solver uses 2D finite volumes for Poisson; its inverse is the **discrete Green operator**, Scharfetter-Gummel fluxes for coupled electron and hole drift diffusion, and a damped Newton solve in `psi`, `ln(n/N0)`, `ln(p/N0)`.

Impact ionization source, added to **both** continuity equations:

`G_II = (alpha_n(|E|)*|Jn| + alpha_p(|E|)*|Jp|)/q`.

Silicon coefficients are from the local Van Overstraeten/de Man law. Electron `alpha_n = 7.03e5 exp(-1.231e6/E) cm^-1`, hole `alpha_p = 1.582e6 exp(-2.036e6/E)` below `4e5 V/cm` and `6.71e5 exp(-1.693e6/E)` above that threshold.

The 2D Poisson source uses `q*(ND-NA+p-n)` and is recomputed from carrier transport. A small-signal two-carrier ionization feedback gain is evaluated through the linearized continuity Green operator for a frozen self-consistent potential.

**The final avalanche-feedback onset voltage is estimated by linear extrapolation of the feedback gain from two converged coupled-impact-ionization *subthreshold* solutions.** It is *not* a direct full-transport, transient or DC runaway breakdown calculation. For the baseline `a=0.5um, NA=1e17`, that extrapolation is about `46.13 V` with the stated coarse model. A separate finer voltage continuation of self-consistent subthreshold states gave gain unity about `46.5 V` (not mesh converged).

**The broader screening contour** uses the zero-impact DD electrostatic fields and two-carrier small-signal gain=1. This is a *screening result*, not the fully avalanche-corrected contour.

## Current result counts

- 56 geometries assessed on `dx=dy=0.1 um` grid.
- 12 designs do not satisfy a 50% centerline depletion criterion at VG=-4 V and VD=0 V; not valid OFF-state blocking designs.
- 24 designs have no certified converged coupled-ionization continuation.
- 5 have long gain-to-unity extrapolation or field above ionization-law range.
- 9 form a more conservative coupled-ionization regression subset.
- 6 additional coupled-ionization data points are above `NA=1e18 cm^-3` and were used for an exploratory (not physically qualified) degree-2 regression.

### Regression performance

- Coupled-impact subthreshold-extrapolated VBD surface, **15 points, degree 2**: in-sample R2 `0.8805`, held-out R2 `0.8504`, held-out RMSE `3.06 V`. HIGH-DOPING CAVEAT: 6 of these points exceed 1e18 cm^-3 and classical nondegenerate contacts are uncertain.
- Two-carrier feedback screening contour, **32 OFF-state points, degree 3**: in-sample R2 `0.9679`, held-out R2 `0.6912`, held-out RMSE `1.76 V`.
- Both are regressions on the numerical models, **not experimental measurements**. R2 is not a validation of absolute breakdown physics.

## Limitations to address before a device specification

1. Local impact-ionization coefficients are extrapolated outside roughly `1.75e5–6.4e5 V/cm` and do not include dead-space/nonlocal effects.
2. No validated velocity saturation, lifetime/SRH generation, bandgap narrowing, Fermi-Dirac degenerate P+ statistics, avalanche thermal runaway, or temperature sweep.
3. Rectangular pillar ends and contact corners are ideally sharp, and peak E and VBD remain grid-dependent. Test at `0.05um`, with rounded corners and fitted *geometry* from the actual mask, before interpreting VBD physically.
4. The full nonlinear source failed to converge in substantial portions of the design plane; those values have NOT been filled with arbitrary breakdown numbers.
5. The carrier-current gain=1 indicator is a screening criterion; physical avalanche breakdown requires continuation of coupled impact sources to onset, breakdown current criterion, and mesh/model convergence.

## Files and reproduction

`jfet_vbd_Ld10_fullavalanche_results.jsonl`: all 56 sweep rows, including failures, carrier depletion and gain trajectories.

`jfet_Ld10_full_ionization_contour_vs_regression.png`: coupled avalanche source subthreshold estimates, *missing points masked*, plus degree-2 regression.

`jfet_Ld10_eh_feedback_screen_contour_regression.png`: broader, two-carrier ionization-gain screening contour plus degree-3 regression.

`jfet_Ld10_full_ionization_fit.json`, `jfet_Ld10_eh_feedback_screen_fit.json`: coefficients and metrics.

`jfet_Ld10_full_ionization_predict.py`, `jfet_Ld10_eh_feedback_screen_predict.py`: ready-to-run polynomial predictors.

`jfet_avalanche_full.py`, `jfet_newton_dd_robust.py`, `jfet_avalanche_test.py`, `jfet_vbd_sweep_full.py`: Python solver and sweep.

`jfet_Ld10_avalanche_regression.py`, `jfet_Ld10_feedback_screen_regression.py`: plotting and polynomial fitting.

### Commands

```bash
pip install numpy scipy matplotlib
python jfet_vbd_sweep_full.py --single --a 0.5 --NA 1e17
python jfet_vbd_sweep_full.py --workers 4
python jfet_Ld10_avalanche_regression.py
python jfet_Ld10_feedback_screen_regression.py
python jfet_Ld10_full_ionization_predict.py
```

All generated potential/field solutions are 2D; there are no fitted electric-field curves. Empirical silicon impact-ionization coefficients are material model inputs, separate from the requested polynomial regression on VBD estimates.
