# Deterministic Window Detector v28

This package contains the deterministic window/opening detector script used for the AB/Stantec Group Home floor plans.

It is script-only:
- No AI/model calls
- No OCR model
- No training
- No manual box drawing
- No room-specific coordinate overrides

The detector uses geometric rules:
- PDF page rendering
- Group Home crop detection from PDF text/title blocks
- Wall-band detection
- Cap/sash stroke detection
- Thin/interior wall cap handling
- Door/wall-space rejection
- Pane-evidence trimming so wall space is not counted as a window
- Two-panel horizontal window cleanup

## Files

```text
window_detector_unified_profiles_v28_parallel_pane_single_rescue.py
requirements.txt
README.md
```

## Install

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Required Python packages:

```text
opencv-python
PyMuPDF
numpy
```

## Input PDF

Place the AB/Stantec PDF in the same folder as the script, or pass the full path to it.

Expected input filename:

```text
AB-2026-02678-Plan No. 024082_Addendum No. 01.pdf
```

## Run Group Home 5 on Page 17

This reproduces the latest Group Home 5 v28 output:

```bash
python window_detector_unified_profiles_v28_parallel_pane_single_rescue.py \
  "AB-2026-02678-Plan No. 024082_Addendum No. 01.pdf" \
  --pages 17 \
  --group-homes 5 \
  --outdir page17_group_home5_v28_parallel_pane_single_rescue_output \
  --save-raw \
  --zip
```

Expected console result:

```text
Page 17 / Group Home 5:
14 filtered candidates from 48 raw wall-band candidates
```

## Run Multiple Group Homes

Example:

```bash
python window_detector_unified_profiles_v28_parallel_pane_single_rescue.py \
  "AB-2026-02678-Plan No. 024082_Addendum No. 01.pdf" \
  --pages 17 \
  --group-homes 1 2 3 4 5 26 \
  --outdir page17_all_group_homes_v28_output \
  --save-raw \
  --zip
```

## Run All Detected Group Homes on a Page

Omit `--group-homes`:

```bash
python window_detector_unified_profiles_v28_parallel_pane_single_rescue.py \
  "AB-2026-02678-Plan No. 024082_Addendum No. 01.pdf" \
  --pages 17 \
  --outdir page17_all_detected_group_homes_v28_output \
  --save-raw \
  --zip
```

## Important Settings

The script renders pages at 200 DPI by default.

Do not change DPI if you want pixel coordinates and annotated images to match prior outputs.

Default:

```bash
--dpi 200
```

Changing DPI changes output pixel coordinates.

## Output Files

For each group home, the script writes:

```text
page17_group_home_5_window_candidates_annotated.png
page17_group_home_5_window_candidates.csv
page17_group_home_5_raw_wall_band_candidates.csv
page17_group_home_5_raw_wall_band_candidates_annotated.png
page17_group_home_5_crop.png
summary.csv
```

With `--zip`, it also creates a zipped copy of the output folder.

## CSV Columns

The candidate CSV includes:

```text
page
group_home
id
x_px
y_px
width_px
height_px
center_x_px
center_y_px
orientation
source
score
exterior_side
```

Coordinates are in rendered crop pixels.

## What v28 Fixes

v28 includes the latest repeatable rules from the review cycle:

```text
- wall-space after a window is not counted as part of the window
- wall-space before a window is not counted as part of the window
- horizontal window candidates must have internal parallel pane/sash evidence
- single-pane horizontal windows are trimmed to the actual pane evidence
- two-panel windows are retained only when two real pane segments are detected
- false wall-space candidates are removed
```

## Notes

This script is calibrated for the Stantec / Alberta Group Home drawing profile in the uploaded AB PDF.

It is deterministic and repeatable for the same:
- PDF
- page number
- group home number
- DPI
- script version

For exact reproduction of the latest shown Group Home 5 output, use v28, page 17, group home 5, and default 200 DPI.
