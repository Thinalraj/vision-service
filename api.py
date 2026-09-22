"""FastAPI interface for one-shot RealSense image and size measurements.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import pyrealsense2 as rs
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

WIDTH, HEIGHT, FPS = 1280, 720, 30
SETTINGS_FILE = Path(__file__).with_name("settings.json")


class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics = None
        self.lock = Lock()

    def frame(self):
        with self.lock:
            frames = self.align.process(self.pipeline.wait_for_frames())
            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                raise HTTPException(503, "Camera frame is unavailable")
            self.intrinsics = depth.profile.as_video_stream_profile().intrinsics
            return np.asanyarray(color.get_data()).copy(), np.asanyarray(depth.get_data()).copy()

    def close(self):
        self.pipeline.stop()


camera = None
latest_image = None


@asynccontextmanager
async def lifespan(app):
    global camera
    camera = Camera()
    yield
    camera.close()


app = FastAPI(title="Vision Service API", version="1.0", lifespan=lifespan)


def object_mask(image, object_type):
    """Build an object mask from the selected mode's saved Lab background."""
    try:
        settings = __import__("json").loads(SETTINGS_FILE.read_text())
        calibration = settings.get("modes", {}).get(object_type, {}).get("color_calibration")
    except (OSError, ValueError):
        calibration = None
    if not calibration:
        raise HTTPException(400, f"No colour calibration saved for type '{object_type}'")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    center = np.asarray(calibration["lab_median"], dtype=np.float32)
    spread = np.asarray(calibration["lab_spread"], dtype=np.float32)
    tolerance = np.maximum(spread * 3.0, np.array([12.0, 10.0, 10.0]))
    mask = (~np.all(np.abs(lab - center) <= tolerance, axis=2)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def measure(image, depth, object_type):
    mask = object_mask(image, object_type)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 100]
    if not contours:
        raise HTTPException(404, "No valid object found")
    contour = max(contours, key=cv2.contourArea)
    pixels = cv2.drawContours(np.zeros_like(mask), [contour], -1, 255, -1)
    valid = depth[(pixels > 0) & (depth > 0)]
    if not valid.size:
        raise HTTPException(404, "No valid depth pixels found")
    z = float(np.median(valid)) * camera.depth_scale
    area = float(cv2.countNonZero(pixels) * z * z /
                 (camera.intrinsics.fx * camera.intrinsics.fy) * 1_000_000)
    result = {"type": object_type, "surface_area_mm2": area,
              "estimated_depth_mm": z * 1000.0}
    if object_type == "coin":
        (x, y), radius = cv2.minEnclosingCircle(contour)
        scale = z * 1000.0 * (1 / camera.intrinsics.fx + 1 / camera.intrinsics.fy) / 2
        result.update(center=[round(x, 1), round(y, 1)], diameter_mm=2 * radius * scale)
    else:
        _, _, w, h = cv2.boundingRect(contour)
        result.update(width_mm=w * z / camera.intrinsics.fx * 1000,
                      length_mm=h * z / camera.intrinsics.fy * 1000)
    return result


@app.get("/image")
def get_image():
    global latest_image
    latest_image, _ = camera.frame()
    ok, encoded = cv2.imencode(".jpg", latest_image)
    if not ok:
        raise HTTPException(500, "Could not encode image")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/size")
def get_size(type: str = Query("coin", min_length=1)):
    global latest_image
    normalized = type.strip().lower()
    supported = {"coin": "Coin", "bar": "Bar", "chain": "Chain",
                 "ring": "Ring", "bangle": "Bangle",
                 "ornament": "Ornaments / Pendants / Others"}
    if normalized not in supported:
        raise HTTPException(400, f"Unsupported type. Use: {', '.join(supported)}")
    latest_image, depth = camera.frame()
    return measure(latest_image, depth, supported[normalized])
