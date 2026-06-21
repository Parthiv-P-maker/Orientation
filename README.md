# Object Orientation Detector

Computer vision application that detects objects from a webcam feed and estimates
their orientation angle with respect to both the horizontal and vertical image axes.

## Features

- Detects one or more objects in an image, video, or webcam feed.
- Estimates each object's major-axis orientation angle using image moments by default.
- Prints both horizontal and vertical orientation angles, plus full axis angle,
  signed angle, confidence, and box size.
- Draws the rotated bounding box, center point, reference line, orientation line, and angle label.
- Supports colored objects by default, with extra modes for dark or bright objects.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run in real time with a webcam (default camera index 0):

```powershell
python src\Object_detection.py --webcam 0
```

The output overlay and console summary will show both the horizontal and vertical
angles for each detected object.

The output overlay and console summary will show both the horizontal and vertical
angles for each detected object.

Run on a video and save an annotated demo:

```powershell
python src\Object_detection.py --source demo.mp4 --output Outputs\demo_result.mp4 --no-display
```

Press `q` or `Esc` to quit a live display window.

## Useful Options

- `--mask-mode color`: Detect saturated colored objects. This is the default.
- `--mask-mode dark`: Detect dark objects on a light background.
- `--mask-mode bright`: Detect bright objects on a dark background.
- `--method moments`: Estimate orientation from the object's image moments. This is the default.
- `--method pca`: Estimate orientation from the contour's principal axis.
- `--method box`: Estimate orientation from the long side of the minimum-area rectangle.
- `--angle-format acute`: Show the smallest angle from horizontal, `0` to `90` degrees.
- `--angle-format axis`: Show the full undirected axis angle, `0` to `180` degrees.
- `--angle-format signed`: Show the signed angle, `-90` to `90` degrees.
- `--hue-min`, `--hue-max`, `--sat-min`, `--value-min`: Tune HSV color segmentation.
- `--min-area 1000`: Ignore contours smaller than this area in pixels.
- `--largest-only`: Keep only the largest detected object.
- `--max-objects N`: Keep only the largest `N` detected objects.
- `--best-only`: Only print the best detection by confidence and highlight it in the overlay.
- `--show-mask`: Show the binary segmentation mask.
- `--no-display`: Run headless and only write output / print results.

## Approach

1. The input frame is blurred slightly to reduce noise.
2. A binary mask is created using HSV saturation for colored objects, or grayscale
   thresholding for dark/bright objects.
3. Morphological open and close operations remove small noise and fill small holes.
4. External contours are extracted from the cleaned mask.
5. Small contours are filtered out using `--min-area`.
6. For each remaining contour, the orientation is estimated using image moments by
   default. This uses the whole object shape, not just one rectangle angle, and is
   usually more stable for filled objects.
7. A confidence score is calculated from how elongated the detected shape is. Long,
   narrow objects have more reliable orientation than circular or square objects.
8. The displayed angle is normalized according to `--angle-format`. The default
   `acute` mode reports the smallest angle from the horizontal reference axis.
9. The output frame is annotated with the bounding box, center point, horizontal
   reference line, orientation axis, confidence, and angle text.

## Demo Video

For the demonstration video, run the webcam command and record the application
window, or process a video file with `--output` to generate an annotated result.

## GitHub Deliverable

Commit the source code, sample images, `requirements.txt`, and this README to your
repository. Include the generated demo video or upload it separately and link it in
the repository description.
