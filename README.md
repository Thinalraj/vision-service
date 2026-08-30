# Vision Service

Manual Intel RealSense D405 measurement for coins, bangles, rings, bars, and chains using aligned depth and color frames.

## Requirements

- Intel RealSense D405 camera
- Python 3
- Intel RealSense SDK and Python bindings

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python diameter.py
```

Choose an object from the opening menu to start the camera, then click two measurement points. Press `R` to reset, `M` to return to the menu, and `Esc` to exit.

The live camera window also includes a clickable `MENU` button in its upper-right corner. It stops the camera cleanly and returns to the object selector.

Press `C` to start colour calibration. Click nine clean background locations spread across the image. Each click samples an 11 x 11 pixel patch; after the ninth point, a robust CIE Lab background profile is saved to `vision_config.json` for the selected object. Press `C` again before completing nine points to cancel.

Press `A` to draw and save an area of interest (AOI) for the selected object. Drag the rectangle and press `Enter` or `Space` to accept it. The main live window then displays only that cropped region while preserving full-frame coordinates for depth and colour sampling. Press `X` to reset all object AOIs to the default full camera frame.

After colour calibration, press `D` to run detection once and lock the resulting mask and outline. Detection is not recalculated on every frame. Press `R` to clear the locked detection and measurements. The mask is cleaned with morphological filtering and the main contour is isolated. Coin mode requires a reasonably circular contour and displays a fitted enclosing circle. Bar mode requires a rectangular contour and displays a rotated fitted rectangle, supporting bars, plates, and square samples at different angles. Press `B` to toggle between the isolated-object and raw camera views without rerunning detection.

Bangle mode keeps the stricter two-boundary detector: it fits the outer contour, then finds and fits the largest elliptical hole inside it. Ring mode is separate and assumes the ring is on a green background. It converts the AOI to HSV, segments the green surface, inverts that mask to find non-green objects, then uses contour hierarchy to find an outer contour with an inner hole. It rejects unlikely candidates by area, maximum expected physical size, circularity, ellipse aspect ratio, inner/outer diameter ratio, and concentricity. The selected candidate reports a confidence score and draws the outer contour, inner contour, fitted ellipses, center point, and pixel diameters. Ring mode keeps the raw camera view visible after detection so dark rings do not disappear into a black isolated-mask display; press `B` to toggle into the isolated mask debug view.

Ring detector thresholds live in `RING_DETECTOR_CONFIG` inside `diameter.py`. You can override any of those values in `vision_config.json` under a `ring_detector` object, for example HSV limits for a different green surface.

Press `S` after detection or a manual two-point measurement to save a result. The first save asks for the CSV destination and stores that path in `vision_config.json`. Every subsequent save asks only for the item name and appends one new row to the same CSV. Press `O` to select a different results file. Each row contains the timestamp, item name, object type, diameter, projected surface area, estimated camera depth, and AOI coordinates. A manual two-point diameter takes priority over the fitted-circle estimate. Bar results also include rotated-rectangle width, length, top surface area, and estimated thickness. Thickness is calculated from the median background depth minus the median object depth inside the AOI.

The application displays the unblended BGR camera stream so its colors match the RealSense Viewer more closely.

Both aligned depth and colour streams run at 1280 x 720 and 30 FPS. Supported RealSense sensors have auto exposure enabled, and the colour sensor has auto white balance enabled. Re-run colour calibration after changing camera resolution or lighting.
