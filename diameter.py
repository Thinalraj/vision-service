import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "D405 Vision Measurement"
MENU_NAME = "Select Measurement Type"
OBJECT_OPTIONS = ["Coin", "Bangle", "Ring", "Bar", "Chain"]
CALIBRATION_FILE = Path(__file__).with_name("vision_config.json")
CALIBRATION_POINT_COUNT = 9
SAMPLE_RADIUS = 5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30
IMAGE_DISPLAY_WIDTH = 960
IMAGE_DISPLAY_HEIGHT = 720
CONTROL_PANEL_WIDTH = 360
APP_DISPLAY_WIDTH = IMAGE_DISPLAY_WIDTH + CONTROL_PANEL_WIDTH
APP_DISPLAY_HEIGHT = IMAGE_DISPLAY_HEIGHT
clicked_points = []
depth_frame_global = None
intrinsics_global = None
selected_object = None
current_color_image = None
calibration_mode = False
calibration_points = []
calibration_samples = []
current_aoi = (0, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1)
active_color_calibration = None
background_removal_enabled = True
latest_result = {}
result_error = ""
menu_requested = False
control_buttons = []
pending_action = None
display_image_rect = (0, 0, IMAGE_DISPLAY_WIDTH - 1, IMAGE_DISPLAY_HEIGHT - 1)
last_detection_debug_reason = ""


def load_config():
    if not CALIBRATION_FILE.exists():
        return {"version": 1, "color_calibrations": {}}
    try:
        with CALIBRATION_FILE.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (OSError, ValueError) as error:
        print("Could not load {}: {}".format(CALIBRATION_FILE, error))
        return {"version": 1, "color_calibrations": {}}


def write_config(config_data):
    with CALIBRATION_FILE.open("w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, indent=2)
        config_file.write("\n")


def load_aoi():
    saved = load_config().get("areas", {}).get(selected_object)
    saved_resolution = [640, 480]
    if isinstance(saved, dict):
        saved_resolution = saved.get("resolution", [FRAME_WIDTH, FRAME_HEIGHT])
        saved = saved.get("coordinates")
    if isinstance(saved, list) and len(saved) == 4:
        scale_x = FRAME_WIDTH / float(saved_resolution[0])
        scale_y = FRAME_HEIGHT / float(saved_resolution[1])
        scaled = (
            int(round(saved[0] * scale_x)),
            int(round(saved[1] * scale_y)),
            int(round(saved[2] * scale_x)),
            int(round(saved[3] * scale_y)),
        )
        return (
            max(0, min(FRAME_WIDTH - 1, scaled[0])),
            max(0, min(FRAME_HEIGHT - 1, scaled[1])),
            max(0, min(FRAME_WIDTH - 1, scaled[2])),
            max(0, min(FRAME_HEIGHT - 1, scaled[3])),
        )
    return (0, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1)


def save_aoi(aoi):
    config_data = load_config()
    config_data.setdefault("areas", {})[selected_object] = {
        "coordinates": list(aoi),
        "resolution": [FRAME_WIDTH, FRAME_HEIGHT],
    }
    write_config(config_data)
    print("Saved {} AOI: {}".format(selected_object, aoi))


def reset_all_aois():
    config_data = load_config()
    config_data["areas"] = {}
    write_config(config_data)
    print("All AOIs reset to the full {} x {} frame.".format(
        FRAME_WIDTH, FRAME_HEIGHT))


def enable_automatic_camera_controls(device):
    """Enable supported automatic exposure and white-balance controls."""
    for sensor in device.query_sensors():
        sensor_name = sensor.get_info(rs.camera_info.name)
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1.0)
            print("Auto exposure enabled on {}.".format(sensor_name))
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
            print("Auto white balance enabled on {}.".format(sensor_name))


def point_in_aoi(x, y):
    left, top, right, bottom = current_aoi
    return left <= x <= right and top <= y <= bottom


def choose_aoi():
    """Pause the live view while the operator draws a new AOI."""
    global current_aoi
    if current_color_image is None:
        return
    selector_name = "Set AOI - drag, then press Enter or Space"
    x, y, width, height = cv2.selectROI(
        selector_name, current_color_image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(selector_name)
    if width > 1 and height > 1:
        current_aoi = (int(x), int(y), int(x + width - 1), int(y + height - 1))
        save_aoi(current_aoi)
    else:
        print("AOI selection cancelled; previous AOI kept.")


def save_color_calibration(samples):
    """Save a robust Lab background model for the selected object mode."""
    global active_color_calibration
    pixels = np.concatenate(samples, axis=0).astype(np.float32)
    center = np.median(pixels, axis=0)
    median_deviation = np.median(np.abs(pixels - center), axis=0)
    spread = np.maximum(median_deviation * 1.4826, 2.0)

    config_data = load_config()
    calibrations = config_data.setdefault("color_calibrations", {})
    calibrations[selected_object] = {
        "color_space": "OpenCV Lab",
        "lab_median": center.round(3).tolist(),
        "lab_spread": spread.round(3).tolist(),
        "sample_points": [list(point) for point in calibration_points],
        "sample_patch_size": SAMPLE_RADIUS * 2 + 1,
        "resolution": [int(current_color_image.shape[1]),
                       int(current_color_image.shape[0])],
    }
    active_color_calibration = calibrations[selected_object]
    write_config(config_data)

    print("Saved {} background calibration to {}".format(
        selected_object, CALIBRATION_FILE))
    print("Lab median:", center.round(2).tolist())
    print("Lab spread:", spread.round(2).tolist())


def add_calibration_point(x, y):
    global calibration_mode
    height, width = current_color_image.shape[:2]
    left, right = max(0, x - SAMPLE_RADIUS), min(width, x + SAMPLE_RADIUS + 1)
    top, bottom = max(0, y - SAMPLE_RADIUS), min(height, y + SAMPLE_RADIUS + 1)
    patch = current_color_image[top:bottom, left:right]
    lab_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    calibration_points.append((x, y))
    calibration_samples.append(lab_patch)
    print("Calibration point {}/{}: ({}, {})".format(
        len(calibration_points), CALIBRATION_POINT_COUNT, x, y))

    if len(calibration_points) == CALIBRATION_POINT_COUNT:
        save_color_calibration(calibration_samples)
        calibration_mode = False
        print("Colour calibration complete. Measurement mode restored.")


def draw_menu():
    """Show the object selector without starting the camera."""
    menu = np.full((520, 720, 3), (32, 35, 42), dtype=np.uint8)
    cv2.putText(menu, "Select an object", (210, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(menu, "The camera starts after your selection", (170, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 190, 200), 1, cv2.LINE_AA)

    button_width, button_height = 220, 85
    positions = [(75, 145), (345, 145), (75, 265), (345, 265), (210, 385)]
    buttons = []
    for label, (x, y) in zip(OBJECT_OPTIONS, positions):
        cv2.rectangle(menu, (x, y), (x + button_width, y + button_height),
                      (72, 119, 210), -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        text_x = x + (button_width - text_size[0]) // 2
        text_y = y + (button_height + text_size[1]) // 2
        cv2.putText(menu, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (255, 255, 255), 2, cv2.LINE_AA)
        buttons.append((label, x, y, x + button_width, y + button_height))
    return menu, buttons


def select_object():
    """Wait for a menu click and return the selection, or None on exit."""
    selection = {"value": None}
    menu, buttons = draw_menu()

    def menu_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for label, left, top, right, bottom in buttons:
                if left <= x <= right and top <= y <= bottom:
                    selection["value"] = label
                    return

    cv2.namedWindow(MENU_NAME)
    cv2.setMouseCallback(MENU_NAME, menu_callback)
    while selection["value"] is None:
        cv2.imshow(MENU_NAME, menu)
        if cv2.waitKey(20) == 27:
            cv2.destroyWindow(MENU_NAME)
            return None
        if cv2.getWindowProperty(MENU_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return None

    cv2.destroyWindow(MENU_NAME)
    return selection["value"]


def mouse_callback(event, x, y, flags, param):
    global clicked_points, latest_result, menu_requested, pending_action
    if event != cv2.EVENT_LBUTTONDOWN or depth_frame_global is None:
        return

    for action, left, top, right, bottom in control_buttons:
        if left <= x <= right and top <= y <= bottom:
            pending_action = action
            return

    image_left, image_top, image_right, image_bottom = display_image_rect
    if not (image_left <= x <= image_right and image_top <= y <= image_bottom):
        return

    image_width = max(1, image_right - image_left + 1)
    image_height = max(1, image_bottom - image_top + 1)
    aoi_width = max(1, current_aoi[2] - current_aoi[0] + 1)
    aoi_height = max(1, current_aoi[3] - current_aoi[1] + 1)
    x = int(round((x - image_left) * aoi_width / image_width)) + current_aoi[0]
    y = int(round((y - image_top) * aoi_height / image_height)) + current_aoi[1]
    x = max(current_aoi[0], min(current_aoi[2], x))
    y = max(current_aoi[1], min(current_aoi[3], y))

    if not point_in_aoi(x, y):
        print("Point is outside the AOI.")
        return

    if calibration_mode and current_color_image is not None:
        add_calibration_point(x, y)
        return

    depth = depth_frame_global.get_distance(x, y)
    if depth <= 0:
        print("Invalid depth at this point.")
        return

    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics_global, [x, y], depth)
    clicked_points.append({"pixel": (x, y), "point": point_3d, "depth": depth})
    print("\nPoint selected:")
    print("Pixel:", x, y)
    print("Depth: {:.3f} mm".format(depth * 1000))
    print("3D: X={:.3f} mm  Y={:.3f} mm  Z={:.3f} mm".format(
        point_3d[0] * 1000, point_3d[1] * 1000, point_3d[2] * 1000))

    if len(clicked_points) > 2:
        clicked_points = clicked_points[-2:]
    if len(clicked_points) == 2:
        p1 = np.array(clicked_points[0]["point"])
        p2 = np.array(clicked_points[1]["point"])
        distance_mm = np.linalg.norm(p2 - p1) * 1000
        latest_result["diameter_mm"] = float(distance_mm)
        print("\n===================================")
        print("Measured {} size:".format(selected_object.lower()))
        print("{:.3f} mm".format(distance_mm))
        print("===================================\n")


def draw_measurement(display):
    offset_x, offset_y = current_aoi[0], current_aoi[1]
    for index, item in enumerate(clicked_points):
        x = item["pixel"][0] - offset_x
        y = item["pixel"][1] - offset_y
        cv2.circle(display, (x, y), 6, (255, 255, 255), -1)
        cv2.putText(display, "P{}".format(index + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

    if len(clicked_points) != 2:
        return

    p1_pixel = (clicked_points[0]["pixel"][0] - offset_x,
                clicked_points[0]["pixel"][1] - offset_y)
    p2_pixel = (clicked_points[1]["pixel"][0] - offset_x,
                clicked_points[1]["pixel"][1] - offset_y)
    cv2.line(display, p1_pixel, p2_pixel, (255, 255, 255), 2)
    p1 = np.array(clicked_points[0]["point"])
    p2 = np.array(clicked_points[1]["point"])
    distance_mm = np.linalg.norm(p2 - p1) * 1000
    midpoint = (int((p1_pixel[0] + p2_pixel[0]) / 2),
                int((p1_pixel[1] + p2_pixel[1]) / 2))
    cv2.putText(display, "{:.2f} mm".format(distance_mm),
                (midpoint[0] - 40, midpoint[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_calibration(display):
    offset_x, offset_y = current_aoi[0], current_aoi[1]
    for index, (x, y) in enumerate(calibration_points):
        x -= offset_x
        y -= offset_y
        cv2.circle(display, (x, y), SAMPLE_RADIUS + 3, (0, 255, 255), 2)
        cv2.putText(display, str(index + 1), (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    cv2.LINE_AA)


def is_ring_like_object():
    return selected_object in ("Bangle", "Ring")


def find_annulus_hole(object_mask, contour, outer_ellipse, area, center_x,
                      center_y, radius):
    outer_fill = np.zeros_like(object_mask)
    cv2.drawContours(outer_fill, [contour], -1, 255, thickness=-1)
    hole_mask = cv2.bitwise_and(outer_fill, cv2.bitwise_not(object_mask))
    hole_contours, _ = cv2.findContours(
        hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_holes = []
    minimum_hole_area = max(20.0, area * 0.015)
    minimum_axis_ratio = 0.25
    minimum_fit = 0.55
    center_limit = 0.45
    nesting_limit = 0.98
    for hole_contour in hole_contours:
        hole_area = cv2.contourArea(hole_contour)
        if hole_area < minimum_hole_area or len(hole_contour) < 5:
            continue
        hole_ellipse = cv2.fitEllipse(hole_contour)
        hole_axis_a, hole_axis_b = hole_ellipse[1]
        hole_major = max(hole_axis_a, hole_axis_b)
        hole_minor = min(hole_axis_a, hole_axis_b)
        if hole_major <= 1 or hole_minor / hole_major < minimum_axis_ratio:
            continue
        hole_ellipse_area = np.pi * hole_axis_a * hole_axis_b * 0.25
        hole_fit = min(hole_area, hole_ellipse_area) / max(
            hole_area, hole_ellipse_area)
        if hole_fit >= minimum_fit:
            valid_holes.append((hole_contour, hole_area, hole_ellipse, hole_fit))

    if not valid_holes:
        return None, None

    inner_contour, _, inner_ellipse, _ = max(
        valid_holes, key=lambda candidate: candidate[1] * candidate[3])
    (inner_x, inner_y), inner_radius = cv2.minEnclosingCircle(inner_contour)
    center_offset = np.hypot(inner_x - center_x, inner_y - center_y)
    outer_axes = sorted(outer_ellipse[1])
    inner_axes = sorted(inner_ellipse[1])
    axes_are_nested = (
        inner_axes[0] < outer_axes[0] * nesting_limit and
        inner_axes[1] < outer_axes[1] * nesting_limit)
    if inner_radius <= 1 or center_offset > radius * center_limit or not axes_are_nested:
        return None, None
    return {
        "center": (int(round(inner_x)), int(round(inner_y))),
        "radius": int(round(inner_radius)),
    }, inner_ellipse


def segment_object(cropped_bgr):
    """Remove the calibrated background and locate the main object contour."""
    global last_detection_debug_reason

    if not active_color_calibration:
        return cropped_bgr, None

    lab_image = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    center = np.array(active_color_calibration["lab_median"], dtype=np.float32)
    spread = np.array(active_color_calibration["lab_spread"], dtype=np.float32)
    # A minimum tolerance prevents sensor noise from defeating a very uniform
    # calibration. Three robust spreads cover normal background variation.
    tolerance = np.maximum(spread * 3.0, np.array([12.0, 10.0, 10.0]))
    background = np.all(np.abs(lab_image - center) <= tolerance, axis=2)
    object_mask = (~background).astype(np.uint8) * 255

    kernel_size = 3 if is_ring_like_object() else 5
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if is_ring_like_object():
        # Closing fills small breaks without eroding a thin ring band away.
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)
    else:
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

    object_pixel_count = cv2.countNonZero(object_mask)
    if object_pixel_count < 20:
        last_detection_debug_reason = "no object pixels after background removal"
        return cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=object_mask), None

    contours, _ = cv2.findContours(
        object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_fraction = 0.0002 if is_ring_like_object() else 0.002
    minimum_area = max(
        30.0 if is_ring_like_object() else 100.0,
        cropped_bgr.shape[0] * cropped_bgr.shape[1] * area_fraction)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < minimum_area or perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        candidates.append((contour, area, circularity))

    if not candidates:
        last_detection_debug_reason = "object found but no valid contour"
        return cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=object_mask), None

    rectangle = None
    rectangularity = None
    outer_ellipse = None
    if selected_object == "Coin":
        circular = [candidate for candidate in candidates if candidate[2] >= 0.55]
        if not circular:
            return cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=object_mask), None
        contour, area, circularity = max(
            circular, key=lambda candidate: candidate[1] * candidate[2])
    elif is_ring_like_object():
        elliptical = []
        minimum_axis_ratio = 0.25
        minimum_fit = 0.60
        for candidate in candidates:
            candidate_contour, candidate_area, candidate_circularity = candidate
            if len(candidate_contour) < 5:
                continue
            candidate_ellipse = cv2.fitEllipse(candidate_contour)
            axis_a, axis_b = candidate_ellipse[1]
            major_axis = max(axis_a, axis_b)
            minor_axis = min(axis_a, axis_b)
            if major_axis <= 1 or minor_axis / major_axis < minimum_axis_ratio:
                continue
            ellipse_area = np.pi * axis_a * axis_b * 0.25
            area_fit = min(candidate_area, ellipse_area) / max(candidate_area, ellipse_area)
            if area_fit >= minimum_fit:
                elliptical.append((
                    candidate_contour, candidate_area, candidate_circularity,
                    candidate_ellipse, area_fit))
        if not elliptical:
            last_detection_debug_reason = "object found but ring/bangle shape invalid"
            return cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=object_mask), None
        contour, area, circularity, outer_ellipse, _ = max(
            elliptical, key=lambda candidate: candidate[1] * candidate[4])
    elif selected_object == "Bar":
        rectangular = []
        for candidate in candidates:
            candidate_contour, candidate_area, candidate_circularity = candidate
            hull = cv2.convexHull(candidate_contour)
            hull_perimeter = cv2.arcLength(hull, True)
            corners = cv2.approxPolyDP(hull, 0.03 * hull_perimeter, True)
            if len(corners) < 4 or len(corners) > 6:
                continue
            candidate_rectangle = cv2.minAreaRect(candidate_contour)
            rect_width, rect_height = candidate_rectangle[1]
            rectangle_area = rect_width * rect_height
            if rectangle_area <= 0:
                continue
            candidate_rectangularity = candidate_area / rectangle_area
            if candidate_rectangularity >= 0.65:
                rectangular.append((
                    candidate_contour,
                    candidate_area,
                    candidate_circularity,
                    candidate_rectangle,
                    candidate_rectangularity,
                ))
        if not rectangular:
            last_detection_debug_reason = "object found but rectangle shape invalid"
            return cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=object_mask), None
        contour, area, circularity, rectangle, rectangularity = max(
            rectangular, key=lambda candidate: candidate[1] * candidate[4])
    else:
        contour, area, circularity = max(candidates, key=lambda candidate: candidate[1])

    selected_mask = np.zeros_like(object_mask)
    cv2.drawContours(selected_mask, [contour], -1, 255, thickness=-1)
    selected_mask = cv2.bitwise_and(selected_mask, object_mask)
    isolated = cv2.bitwise_and(cropped_bgr, cropped_bgr, mask=selected_mask)
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    inner_circle = None
    inner_ellipse = None
    if is_ring_like_object():
        inner_circle, inner_ellipse = find_annulus_hole(
            object_mask, contour, outer_ellipse, area, center_x, center_y,
            radius)
        if inner_circle is None:
            last_detection_debug_reason = "object found but inner hole invalid"
            return isolated, None
    detection = {
        "detected": True,
        "contour": contour,
        "center": (int(round(center_x)), int(round(center_y))),
        "radius": int(round(radius)),
        "circularity": float(circularity),
        "rectangle": None,
        "rectangularity": rectangularity,
        "mask": selected_mask,
        "inner_circle": inner_circle,
        "outer_ellipse": outer_ellipse,
        "inner_ellipse": inner_ellipse,
        "score": 1.0,
        "inner_detected": inner_circle is not None,
    }
    if rectangle is not None:
        detection["rectangle"] = np.int32(cv2.boxPoints(rectangle))
    return isolated, detection


def draw_detection(display, detection):
    if detection is None:
        return
    cv2.drawContours(display, [detection["contour"]], -1, (255, 0, 255), 2)
    if selected_object == "Coin" and detection["radius"] > 1:
        cv2.circle(display, detection["center"], detection["radius"],
                   (0, 255, 0), 2)
    if is_ring_like_object() and detection["outer_ellipse"] is not None:
        cv2.ellipse(display, detection["outer_ellipse"], (0, 255, 0), 2)
        if detection.get("inner_ellipse") is not None:
            cv2.ellipse(display, detection["inner_ellipse"], (0, 200, 255), 2)
        if detection.get("inner_contour") is not None:
            cv2.drawContours(display, [detection["inner_contour"]], -1,
                             (0, 200, 255), 2)
        cv2.circle(display, detection["center"], 4, (255, 255, 255), -1)
    if selected_object == "Bar" and detection["rectangle"] is not None:
        cv2.drawContours(display, [detection["rectangle"]], -1, (0, 255, 0), 2)


def format_metric(value, unit="", precision=2):
    if value == "" or value is None:
        return "-"
    try:
        return "{:.{}f}{}".format(float(value), precision, unit)
    except (TypeError, ValueError):
        return str(value)


def draw_panel_text(panel, text, x, y, scale=0.5, color=(220, 225, 230),
                    thickness=1):
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_control_button(panel, label, action, x, y, width, height,
                        enabled=True):
    global control_buttons
    color = (68, 102, 145) if enabled else (55, 60, 65)
    text_color = (255, 255, 255) if enabled else (140, 145, 150)
    cv2.rectangle(panel, (x, y), (x + width, y + height), color, -1)
    cv2.rectangle(panel, (x, y), (x + width, y + height), (175, 185, 195), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
    text_x = x + max(6, (width - text_size[0]) // 2)
    text_y = y + max(text_size[1] + 6, (height + text_size[1]) // 2)
    draw_panel_text(panel, label, text_x, text_y, 0.48, text_color)
    if enabled:
        control_buttons.append((
            action,
            IMAGE_DISPLAY_WIDTH + x,
            y,
            IMAGE_DISPLAY_WIDTH + x + width,
            y + height,
        ))


def result_lines_for_panel():
    if result_error:
        return [("Error", result_error)]
    if not latest_result:
        return [("Result", "No measurement yet")]
    if is_ring_like_object():
        return [
            ("Result", selected_object),
            ("Inner", latest_result.get("inner_status", "-")),
            ("Outer ratio", format_metric(
                latest_result.get("outer_eccentricity_ratio"), "",
                precision=3)),
            ("Inner ratio", format_metric(
                latest_result.get("inner_eccentricity_ratio"), "",
                precision=3)),
            ("Outer major", format_metric(
                latest_result.get("outer_major_axis_mm"), " mm")),
            ("Outer minor", format_metric(
                latest_result.get("outer_minor_axis_mm"), " mm")),
            ("Outer dia", format_metric(
                latest_result.get("outer_diameter_mm"), " mm")),
            ("Inner major", format_metric(
                latest_result.get("inner_major_axis_mm"), " mm")),
            ("Inner minor", format_metric(
                latest_result.get("inner_minor_axis_mm"), " mm")),
            ("Inner dia", format_metric(
                latest_result.get("inner_diameter_mm"), " mm")),
            ("Area", format_metric(
                latest_result.get("surface_area_mm2"), " mm2")),
            ("Depth", format_metric(
                latest_result.get("estimated_depth_mm"), " mm")),
            ("Score", format_metric(
                latest_result.get("confidence_score"), "", precision=3)),
        ]
    if selected_object == "Bar":
        return [
            ("Result", "Bar"),
            ("Width", format_metric(latest_result.get("width_mm"), " mm")),
            ("Length", format_metric(latest_result.get("length_mm"), " mm")),
            ("Area", format_metric(
                latest_result.get("surface_area_mm2"), " mm2")),
            ("Thickness", format_metric(
                latest_result.get("estimated_thickness_mm"), " mm")),
            ("Depth", format_metric(
                latest_result.get("estimated_depth_mm"), " mm")),
        ]
    return [
        ("Result", selected_object),
        ("Diameter", format_metric(latest_result.get("diameter_mm"), " mm")),
        ("Area", format_metric(latest_result.get("surface_area_mm2"), " mm2")),
        ("Depth", format_metric(latest_result.get("estimated_depth_mm"), " mm")),
    ]


def draw_control_panel(instruction, locked_detection):
    global control_buttons
    control_buttons = []
    panel = np.full((APP_DISPLAY_HEIGHT, CONTROL_PANEL_WIDTH, 3),
                    (31, 35, 42), dtype=np.uint8)

    draw_panel_text(panel, "Vision Service", 18, 34, 0.72, (255, 255, 255), 2)
    draw_panel_text(panel, "Mode: {}".format(selected_object), 18, 64,
                    0.52, (180, 210, 255))
    draw_panel_text(panel, instruction, 18, 90, 0.43, (200, 205, 210))

    button_width = 148
    button_height = 42
    left_a = 18
    left_b = 190
    y = 120
    draw_control_button(panel, "Detect", "detect", left_a, y, button_width,
                        button_height)
    draw_control_button(panel, "Reset", "reset", left_b, y, button_width,
                        button_height)
    y += 54
    draw_control_button(panel, "Set AOI", "aoi", left_a, y, button_width,
                        button_height)
    draw_control_button(panel, "Default AOI", "default_aoi", left_b, y,
                        button_width, button_height)
    y += 54
    draw_control_button(panel, "Calibrate", "calibrate", left_a, y,
                        button_width, button_height)
    draw_control_button(panel, "Toggle View", "toggle_view", left_b, y,
                        button_width, button_height,
                        enabled=locked_detection is not None)
    y += 54
    draw_control_button(panel, "Save Result", "save", left_a, y,
                        button_width, button_height)
    draw_control_button(panel, "Output File", "output", left_b, y,
                        button_width, button_height)
    y += 54
    draw_control_button(panel, "Menu", "menu", left_a, y, button_width,
                        button_height)
    draw_control_button(panel, "Exit", "exit", left_b, y, button_width,
                        button_height)

    y += 76
    draw_panel_text(panel, "Measurements", 18, y, 0.58, (255, 255, 255), 2)
    y += 24
    for label, value in result_lines_for_panel():
        draw_panel_text(panel, label, 18, y, 0.45, (185, 195, 205))
        draw_panel_text(panel, str(value), 136, y, 0.45, (0, 255, 255))
        y += 22

    if last_detection_debug_reason:
        y = min(y + 12, APP_DISPLAY_HEIGHT - 74)
        draw_panel_text(panel, "Detector", 18, y, 0.48, (255, 255, 255), 1)
        y += 22
        draw_panel_text(panel, last_detection_debug_reason[:36], 18, y,
                        0.40, (235, 210, 110))
    return panel


def compose_app_frame(camera_display, instruction, locked_detection):
    global display_image_rect
    image = np.full((IMAGE_DISPLAY_HEIGHT, IMAGE_DISPLAY_WIDTH, 3),
                    (12, 14, 18), dtype=np.uint8)
    source_height, source_width = camera_display.shape[:2]
    scale = min(
        IMAGE_DISPLAY_WIDTH / max(1, source_width),
        IMAGE_DISPLAY_HEIGHT / max(1, source_height),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(camera_display, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA)
    left = (IMAGE_DISPLAY_WIDTH - resized_width) // 2
    top = (IMAGE_DISPLAY_HEIGHT - resized_height) // 2
    image[top:top + resized_height, left:left + resized_width] = resized
    panel = draw_control_panel(instruction, locked_detection)
    display_image_rect = (
        left,
        top,
        left + resized_width - 1,
        top + resized_height - 1,
    )
    return np.hstack((image, panel))


def should_show_isolated_detection():
    return background_removal_enabled


def calculate_detection_metrics(detection, depth_frame, intrinsics, depth_scale):
    """Estimate depth, projected area, and circle diameter from a locked mask."""
    left, top, right, bottom = current_aoi
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_crop = depth_raw[top:bottom + 1, left:right + 1]
    valid_depth = depth_crop[(detection["mask"] > 0) & (depth_crop > 0)]
    if valid_depth.size == 0:
        return {}

    depth_m = float(np.median(valid_depth)) * depth_scale
    pixel_area = float(cv2.countNonZero(detection["mask"]))
    surface_area_mm2 = (
        pixel_area * depth_m * depth_m / (intrinsics.fx * intrinsics.fy) * 1_000_000.0
    )
    metrics = {
        "estimated_depth_mm": depth_m * 1000.0,
        "surface_area_mm2": surface_area_mm2,
    }
    if selected_object == "Coin":
        scale_mm_per_pixel = (
            depth_m / intrinsics.fx + depth_m / intrinsics.fy
        ) * 0.5 * 1000.0
        metrics["diameter_mm"] = 2.0 * detection["radius"] * scale_mm_per_pixel
        metrics["outer_diameter_mm"] = metrics["diameter_mm"]
    if is_ring_like_object() and detection["outer_ellipse"] is not None:
        scale_mm_per_pixel = (
            depth_m / intrinsics.fx + depth_m / intrinsics.fy
        ) * 0.5 * 1000.0
        outer_axes_mm = sorted(
            axis * scale_mm_per_pixel for axis in detection["outer_ellipse"][1])
        metrics["outer_minor_axis_mm"] = outer_axes_mm[0]
        metrics["outer_major_axis_mm"] = outer_axes_mm[1]
        metrics["outer_eccentricity_ratio"] = (
            outer_axes_mm[0] / outer_axes_mm[1] if outer_axes_mm[1] > 0 else "")
        metrics["outer_diameter_mm"] = float(np.sqrt(outer_axes_mm[0] * outer_axes_mm[1]))
        metrics["diameter_mm"] = metrics["outer_diameter_mm"]
        metrics["confidence_score"] = detection.get("score", "")
        metrics["inner_status"] = (
            "detected" if detection.get("inner_detected") else "not detected")
        if "outer_diameter_px" in detection:
            metrics["outer_diameter_px"] = detection["outer_diameter_px"]
        if detection.get("inner_detected"):
            metrics["inner_diameter_px"] = detection["inner_diameter_px"]
        if detection.get("inner_ellipse") is not None:
            inner_axes_mm = sorted(
                axis * scale_mm_per_pixel for axis in detection["inner_ellipse"][1])
            metrics["inner_minor_axis_mm"] = inner_axes_mm[0]
            metrics["inner_major_axis_mm"] = inner_axes_mm[1]
            metrics["inner_eccentricity_ratio"] = (
                inner_axes_mm[0] / inner_axes_mm[1] if inner_axes_mm[1] > 0 else "")
            metrics["inner_diameter_mm"] = float(
                np.sqrt(inner_axes_mm[0] * inner_axes_mm[1]))
            metrics["surface_area_mm2"] = (
                np.pi * 0.25 *
                (outer_axes_mm[0] * outer_axes_mm[1] -
                 inner_axes_mm[0] * inner_axes_mm[1])
            )
        else:
            metrics["surface_area_mm2"] = ""
    if selected_object == "Bar" and detection["rectangle"] is not None:
        box = detection["rectangle"].astype(np.float32)
        side_lengths_mm = []
        for index in range(4):
            point_a = box[index]
            point_b = box[(index + 1) % 4]
            delta_x_mm = (point_b[0] - point_a[0]) * depth_m / intrinsics.fx * 1000.0
            delta_y_mm = (point_b[1] - point_a[1]) * depth_m / intrinsics.fy * 1000.0
            side_lengths_mm.append(float(np.hypot(delta_x_mm, delta_y_mm)))
        unique_sides = [
            (side_lengths_mm[0] + side_lengths_mm[2]) * 0.5,
            (side_lengths_mm[1] + side_lengths_mm[3]) * 0.5,
        ]
        width_mm, length_mm = sorted(unique_sides)
        metrics["width_mm"] = width_mm
        metrics["length_mm"] = length_mm
        metrics["surface_area_mm2"] = width_mm * length_mm

        background_depth = depth_crop[
            (detection["mask"] == 0) & (depth_crop > 0)]
        if background_depth.size > 0:
            background_depth_m = float(np.median(background_depth)) * depth_scale
            metrics["estimated_thickness_mm"] = max(
                0.0, (background_depth_m - depth_m) * 1000.0)
    return metrics


def choose_results_file(parent):
    from tkinter import filedialog

    csv_path = filedialog.asksaveasfilename(
        parent=parent,
        title="Select measurement results file",
        initialfile="measurement_results.csv",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if csv_path:
        config_data = load_config()
        config_data["results_file"] = csv_path
        write_config(config_data)
        print("Results file set to {}".format(csv_path))
    return csv_path


def choose_results_file_dialog():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    choose_results_file(root)
    root.destroy()


def save_result_dialog():
    """Ask for an item name and append the result to the configured CSV."""
    if not latest_result:
        print("No result to save. Detect an object or measure two points first.")
        return

    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    item_name = simpledialog.askstring(
        "Save measurement", "Item name:", parent=root)
    if not item_name:
        root.destroy()
        print("Save cancelled.")
        return

    csv_path = load_config().get("results_file")
    if not csv_path:
        csv_path = choose_results_file(root)
    root.destroy()
    if not csv_path:
        print("Save cancelled.")
        return

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "item_name": item_name,
        "object_type": selected_object,
        "diameter_mm": latest_result.get("diameter_mm", ""),
        "outer_diameter_mm": latest_result.get("outer_diameter_mm", ""),
        "inner_diameter_mm": latest_result.get("inner_diameter_mm", ""),
        "outer_major_axis_mm": latest_result.get("outer_major_axis_mm", ""),
        "outer_minor_axis_mm": latest_result.get("outer_minor_axis_mm", ""),
        "outer_eccentricity_ratio": latest_result.get(
            "outer_eccentricity_ratio", ""),
        "inner_major_axis_mm": latest_result.get("inner_major_axis_mm", ""),
        "inner_minor_axis_mm": latest_result.get("inner_minor_axis_mm", ""),
        "inner_eccentricity_ratio": latest_result.get(
            "inner_eccentricity_ratio", ""),
        "inner_status": latest_result.get("inner_status", ""),
        "outer_diameter_px": latest_result.get("outer_diameter_px", ""),
        "inner_diameter_px": latest_result.get("inner_diameter_px", ""),
        "confidence_score": latest_result.get("confidence_score", ""),
        "width_mm": latest_result.get("width_mm", ""),
        "length_mm": latest_result.get("length_mm", ""),
        "surface_area_mm2": latest_result.get("surface_area_mm2", ""),
        "estimated_depth_mm": latest_result.get("estimated_depth_mm", ""),
        "estimated_thickness_mm": latest_result.get("estimated_thickness_mm", ""),
        "aoi_x1": current_aoi[0],
        "aoi_y1": current_aoi[1],
        "aoi_x2": current_aoi[2],
        "aoi_y2": current_aoi[3],
    }
    fieldnames = list(row.keys())
    csv_file_path = Path(csv_path)
    has_existing_data = csv_file_path.exists() and csv_file_path.stat().st_size > 0
    if has_existing_data:
        with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames
        # Upgrade an older schema once. Normal saves use append mode.
        if existing_fields != fieldnames:
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not has_existing_data:
            writer.writeheader()
        writer.writerow(row)
    print("Saved result for '{}' to {}".format(item_name, csv_path))


def run_camera():
    global depth_frame_global, intrinsics_global, current_color_image
    global calibration_mode, current_aoi, active_color_calibration
    global background_removal_enabled, latest_result, menu_requested
    global last_detection_debug_reason, pending_action, result_error
    current_aoi = load_aoi()
    active_color_calibration = load_config().get(
        "color_calibrations", {}).get(selected_object)
    background_removal_enabled = True
    latest_result = {}
    result_error = ""
    menu_requested = False
    pending_action = None
    last_detection_debug_reason = ""
    locked_detection = None
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT,
                         rs.format.z16, FRAME_RATE)
    # OpenCV expects BGR, so request that channel order directly from RealSense.
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT,
                         rs.format.bgr8, FRAME_RATE)
    profile = pipeline.start(config)
    enable_automatic_camera_controls(profile.get_device())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    print("Starting camera for {} measurement...".format(selected_object.lower()))
    for _ in range(30):
        pipeline.wait_for_frames()
    print("Camera ready.")
    print("A = Set AOI, X = Reset all AOIs, C = Colour calibration.")
    print("D = Detect once, B = Toggle view, S = Save, O = Choose results file.")
    print("Click two points. R = Reset, M = Menu, ESC = Exit.")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    try:
        while True:
            aligned_frames = align.process(pipeline.wait_for_frames())
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            intrinsics_global = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_frame_global = depth_frame

            # The old JET depth blend changed every visible color. Display the
            # unblended BGR frame to match RealSense Viewer color rendering.
            current_color_image = np.asanyarray(color_frame.get_data()).copy()
            left, top, right, bottom = current_aoi
            # Keep processing input separate from the annotated display image.
            raw_cropped = current_color_image[top:bottom + 1,
                                              left:right + 1].copy()
            if calibration_mode:
                display = raw_cropped.copy()
                draw_calibration(display)
                instruction = "CALIBRATE: click background points {}/{}".format(
                    len(calibration_points), CALIBRATION_POINT_COUNT)
            else:
                if should_show_isolated_detection() and locked_detection is not None:
                    display = cv2.bitwise_and(
                        raw_cropped, raw_cropped, mask=locked_detection["mask"])
                    draw_detection(display, locked_detection)
                else:
                    display = raw_cropped.copy()
                    if locked_detection is not None:
                        draw_detection(display, locked_detection)
                draw_measurement(display)
                if locked_detection is not None:
                    instruction = "Mode: {} | Detection locked".format(selected_object)
                elif not active_color_calibration:
                    instruction = "Mode: {} | Press C to calibrate background".format(
                        selected_object)
                else:
                    instruction = "Mode: {} | Press D to detect".format(selected_object)
            app_frame = compose_app_frame(display, instruction, locked_detection)
            cv2.imshow(WINDOW_NAME, app_frame)

            key = cv2.waitKey(1) & 0xFF
            if menu_requested:
                return True
            action = pending_action
            pending_action = None
            if key in (ord("d"), ord("D")):
                action = "detect"
            elif key in (ord("r"), ord("R")):
                action = "reset"
            elif key in (ord("a"), ord("A")):
                action = "aoi"
            elif key in (ord("x"), ord("X")):
                action = "default_aoi"
            elif key in (ord("b"), ord("B")):
                action = "toggle_view"
            elif key in (ord("c"), ord("C")):
                action = "calibrate"
            elif key in (ord("m"), ord("M")):
                action = "menu"
            elif key in (ord("s"), ord("S")):
                action = "save"
            elif key in (ord("o"), ord("O")):
                action = "output"
            elif key == 27:
                action = "exit"

            if action == "exit":
                return False
            if action == "menu":
                return True
            if action == "reset":
                clicked_points.clear()
                locked_detection = None
                latest_result = {}
                result_error = ""
                last_detection_debug_reason = ""
                print("Measurement and detection reset.")
            if action == "aoi":
                calibration_mode = False
                clicked_points.clear()
                calibration_points.clear()
                calibration_samples.clear()
                locked_detection = None
                latest_result = {}
                result_error = ""
                last_detection_debug_reason = ""
                choose_aoi()
                cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
            if action == "default_aoi":
                reset_all_aois()
                current_aoi = (0, 0, FRAME_WIDTH - 1, FRAME_HEIGHT - 1)
                clicked_points.clear()
                locked_detection = None
                latest_result = {}
                result_error = ""
                last_detection_debug_reason = ""
            if action == "detect":
                if not active_color_calibration:
                    result_error = "Calibrate background first"
                    print("No colour calibration. Press C and sample 9 background points first.")
                else:
                    _, new_detection = segment_object(raw_cropped)
                    if new_detection is None:
                        if "no object pixels" in last_detection_debug_reason:
                            result_error = "No Object was placed"
                        else:
                            result_error = "No valid object found"
                        latest_result = {}
                        print("No valid {} object detected.".format(selected_object.lower()))
                    else:
                        manual_diameter = latest_result.get("diameter_mm")
                        locked_detection = new_detection
                        result_error = ""
                        last_detection_debug_reason = "accepted {}".format(
                            selected_object.lower())
                        latest_result = calculate_detection_metrics(
                            new_detection, depth_frame, intrinsics_global, depth_scale)
                        if not latest_result:
                            result_error = "No valid object found"
                            locked_detection = None
                            print("No valid depth pixels for detected object.")
                            continue
                        if manual_diameter is not None:
                            latest_result["diameter_mm"] = manual_diameter
                        background_removal_enabled = True
                        print("{} detection locked. Press R to clear it.".format(
                            selected_object))
                        print("Estimated result:", latest_result)
            if action == "toggle_view":
                background_removal_enabled = not background_removal_enabled
                print("Background removal {}.".format(
                    "enabled" if background_removal_enabled else "disabled"))
            if action == "calibrate":
                calibration_mode = not calibration_mode
                calibration_points.clear()
                calibration_samples.clear()
                locked_detection = None
                latest_result = {}
                result_error = ""
                last_detection_debug_reason = ""
                if calibration_mode:
                    clicked_points.clear()
                    print("Colour calibration started.")
                    print("Click 9 clean background locations spread across the image.")
                else:
                    print("Colour calibration cancelled.")
            if action == "save":
                save_result_dialog()
            if action == "output":
                choose_results_file_dialog()
    finally:
        depth_frame_global = None
        intrinsics_global = None
        current_color_image = None
        calibration_mode = False
        menu_requested = False
        pending_action = None
        last_detection_debug_reason = ""
        pipeline.stop()
        cv2.destroyWindow(WINDOW_NAME)


def main():
    global selected_object
    while True:
        clicked_points.clear()
        selected_object = select_object()
        if selected_object is None:
            break
        print("Selected:", selected_object)
        if not run_camera():
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
