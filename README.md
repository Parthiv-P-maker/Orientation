# Reference-Based Object Orientation Measurement System

A computer vision system that measures object orientation **relative to a stored reference frame**, not relative to the camera's image axes.

Traditional orientation detectors report angles from the camera X-axis. This system stores a reference orientation on the first frame (or user command) and reports all subsequent angles as **relative rotation from that reference**.

## Architecture

```
src/orient/
├── main.py              # CLI entry point
├── config.py            # JSON configuration management
├── detector.py          # Object detection pipeline
├── segmentation.py      # Pluggable segmentation (HSV, threshold, adaptive, edge, background)
├── orientation.py       # Orientation estimators (moments, PCA, min-area-rect, ellipse, hull)
├── reference_manager.py # Reference frame storage and relative angle computation
├── tracker.py           # Centroid-based object tracking
├── comparison.py        # Image-to-image comparison
├── visualization.py     # Professional overlays and dashboard
├── logger.py            # CSV/JSON data export
├── geometry.py          # Angle normalization utilities
├── image_mode.py        # Image vs Image mode
├── video_mode.py        # Video analysis mode
└── live_mode.py         # Live camera mode
```

## Pipeline

```
Frame → Segmentation → Contour Extraction → Orientation Estimation
                                                     ↓
                                            Reference Manager
                                                     ↓
                                        Relative Angle = Current − Reference
                                                     ↓
                                          Tracking → Visualization → Export
```

## Operating Modes

### 1. Image vs Image

Compare two images — the first is the reference, the second is the comparison target.

```powershell
python -m orient.main --mode image --reference samples/ref.jpg --source samples/current.jpg --output outputs/
```

### 2. Video Analysis

The first frame becomes the reference. Tracks objects and reports relative rotation throughout.

```powershell
python -m orient.main --mode video --source video.mp4 --output-video outputs/annotated.mp4 --save-csv --save-json
```

### 3. Live Camera

Press **SPACE** to set the current frame as reference. Press **R** to reset. Press **Q** or **ESC** to quit.

```powershell
python -m orient.main --mode live --webcam 0
```

## Orientation Algorithms

| Method | Flag | Best for |
|--------|------|----------|
| Image Moments | `--method moments` | Filled, solid objects (default) |
| PCA | `--method pca` | Point clouds, sparse contours |
| Min-Area Rectangle | `--method box` | Rectangular objects |
| Ellipse Fitting | `--method ellipse` | Elliptical/rounded objects |
| Convex Hull PCA | `--method hull` | Irregular shapes |

## Segmentation Modes

| Mode | Flag | Description |
|------|------|-------------|
| HSV Color | `--mask-mode color` | Saturated colored objects (default) |
| Dark Object | `--mask-mode dark` | Dark on light background |
| Bright Object | `--mask-mode bright` | Light on dark background |
| Adaptive | `--mask-mode adaptive` | Variable lighting |
| Edge | `--mask-mode edge` | Edge-based detection |
| Background Sub | `--mask-mode background` | Moving objects |

## Industrial Inspection

Check if objects are within angular tolerance:

```powershell
python -m orient.main --mode live --inspection --target-angle 45 --tolerance 2
```

Objects are highlighted **green** (PASS) or **red** (FAIL).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

```powershell
pip install pytest
pytest
```

## CLI Reference

```
--mode            image | video | live
--reference       Reference image (image mode)
--source          Input image or video
--webcam          Camera index (default: 0)
--output          Output directory (default: outputs/)
--output-video    Path for annotated output video
--mask-mode       color | dark | bright | adaptive | edge | background
--method          moments | pca | box | ellipse | hull
--hue-min/max     HSV hue range (0-179)
--sat-min         Minimum saturation (0-255)
--value-min       Minimum value (0-255)
--min-area        Minimum contour area (default: 1000)
--max-objects     Keep only N largest objects
--largest-only    Keep only the largest object
--decimals        Decimal places for angles (default: 2)
--save-csv        Export frame data to CSV
--save-json       Export frame data to JSON
--show-debug      Show binary mask window
--no-display      Run headless
--inspection      Enable pass/fail tolerance checking
--target-angle    Target angle for inspection (default: 0)
--tolerance       Tolerance in degrees (default: 2)
--config          Load settings from JSON file
--save-config     Save settings to JSON file
```

## Mathematical Foundation

The relative angle is computed as:

```
θ_relative = normalize(θ_current − θ_reference)
```

Where `normalize` maps the result to (−180°, 180°]. Positive values indicate counter-clockwise rotation from the reference; negative values indicate clockwise.

Orientation confidence is the eigenvalue ratio:

```
confidence = (λ_major − λ_minor) / λ_major
```

A perfectly circular object has confidence 0 (ambiguous orientation). A long, narrow object approaches confidence 1.

## Applications

- Quality inspection on assembly lines
- Robotic pick-and-place alignment verification
- Part rotation measurement in manufacturing
- Weld seam angle verification
- PCB component orientation checking
