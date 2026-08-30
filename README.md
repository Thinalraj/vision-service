# Vision Service

Manual Intel RealSense D405 coin-diameter measurement using aligned depth and color frames.

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

Click two opposite edges of a coin to measure its 3D diameter. Press `R` to reset and `Esc` to exit.
