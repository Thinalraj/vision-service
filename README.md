# Vision Service

Manual Intel RealSense D405 measurement for coins, rings, bars, and chains using aligned depth and color frames.

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

Press `A` to draw and save an area of interest (AOI) for the selected object. Drag the rectangle and press `Enter` or `Space` to accept it. The main live window then displays only that cropped region while preserving full-frame coordinates for depth and colour sampling. Press `X` to reset all four object AOIs to the default full camera frame.

After colour calibration, press `D` to run detection once and lock the resulting mask and outline. Detection is not recalculated on every frame. Press `R` to clear the locked detection and measurements. The mask is cleaned with morphological filtering and the main contour is isolated. Coin and Ring modes require a reasonably circular contour and display a fitted enclosing circle. Bar mode requires a rectangular contour and displays a rotated fitted rectangle, supporting bars, plates, and square samples at different angles. Press `B` to toggle between the isolated-object and raw camera views without rerunning detection.

Ring mode uses two boundary passes: it fits the outer circular contour, then finds and fits the largest circular hole inside it. Results include outer diameter, inner diameter, and annular top area calculated as `pi / 4 x (outer diameter squared - inner diameter squared)`.

Press `S` after detection or a manual two-point measurement to save a result. The first save asks for the CSV destination and stores that path in `vision_config.json`. Every subsequent save asks only for the item name and appends one new row to the same CSV. Press `O` to select a different results file. Each row contains the timestamp, item name, object type, diameter, projected surface area, estimated camera depth, and AOI coordinates. A manual two-point diameter takes priority over the fitted-circle estimate. Bar results also include rotated-rectangle width, length, top surface area, and estimated thickness. Thickness is calculated from the median background depth minus the median object depth inside the AOI.

The application displays the unblended BGR camera stream so its colors match the RealSense Viewer more closely.

Both aligned depth and colour streams run at 1280 x 720 and 30 FPS. Supported RealSense sensors have auto exposure enabled, and the colour sensor has auto white balance enabled. Re-run colour calibration after changing camera resolution or lighting.
