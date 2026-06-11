# A/B Window Detector Comparison

This repo now has two Stantec/AB detector paths:

- `detectionscript/window_detector_unified_profiles_v28_parallel_pane_single_rescue.py` is the untouched standalone control.
- `app/services/stantec_detector_core.py` is the merged app backend used by FastAPI and the comparison harness.

Run an A/B comparison with:

```powershell
python scratch/ab_compare_detectors.py "path\to\AB-2026-02678-Plan No. 024082_Addendum No. 01.pdf" --pages 17 --group-homes 1 2 3 4 5 26
```

Outputs are written to `scratch/ab_window_detector_output/` by default:

- `standalone/` contains the standalone script CSV/PNG output plus full-page JSON exports.
- `merged/` contains merged backend JSON exports.
- `page<N>_comparison_overlay.png` draws standalone boxes in green and merged boxes in red.
- `page<N>_scorecard.json` and `summary_scorecard.csv` compare merged output against standalone output using IoU matching.

Use `--skip-standalone-run` to reuse existing standalone output in the same outdir.
