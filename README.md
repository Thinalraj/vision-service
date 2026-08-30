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

Press `C` to start colour calibration. Click nine clean background locations spread across the image. Each click samples an 11 x 11 pixel patch; after the ninth point, a robust CIE Lab background profile is saved to `vision_config.json` for the selected object. Press `C` again before completing nine points to cancel.

The application displays the unblended BGR camera stream so its colors match the RealSense Viewer more closely.
