# jfet_Ld10 two-carrier avalanche bundle — review & fixes

Reviewed and run on Windows (Python 3.13, numpy 2.5, scipy 1.18) on 2026-10-25.

## Verdict
The numerical model is **correct** and reproduces its own documented results:
- Baseline `a=0.5, NA=1e17` → **VBD = 46.13 V** (matches README).
- Avalanche regression: degree 2, R²=0.8805, held-out R²=0.8504; status counts
  {not-off:12, long-extrap:5, numeric-fail:24, reliable:9, high-doping:6} (matches README).
- Feedback-screen regression: 32 points, degree 3, R²=0.9679, held-out R²=0.6912,
  RMSE 1.76 V (matches README).

No physics bug was found. The Scharfetter–Gummel transport, Van Overstraeten–de Man
impact-ionization source, and two-carrier small-signal feedback gain are sound.

## Bugs found (portability / packaging) and fixes applied

1. **`jfet_Ld10_feedback_screen_regression.py` crashed on import** — it read its input at
   module level from a hardcoded `/mnt/data` path with no CLI override
   (`FileNotFoundError` on any machine without that exact path).
   **Fix:** added `--input` / `--prefix` argparse options defaulting to the script directory.

2. **`jfet_vbd_sweep_full.py --workers` failed on Windows** — `_worker` used
   `signal.SIGALRM` / `signal.alarm`, which do not exist on Windows, so every geometry
   returned `stage='exception'`.
   **Fix:** guarded the alarm with `hasattr(signal, 'SIGALRM')` (per-geometry timeout is
   simply skipped where unavailable).

3. **Hardcoded Linux `/mnt/data` paths** in `jfet_avalanche_test.py` (`sys.path` insert and
   `np.savez` target), the `--output` default of `jfet_vbd_sweep_full.py`, and the `ROOT`
   defaults of both regression scripts.
   **Fix:** all now default to the script's own directory (`os.path.dirname(__file__)`).

With these changes every README command runs as-is on Windows and Linux.
Only the four source files above were modified; the solver physics is untouched.
