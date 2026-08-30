import numpy as np
import cv2
import math

import sys

import pyrealsense2 as rs

# ---------------------------------------------------------
# Global variables
# ---------------------------------------------------------
clicked_points = []
depth_frame_global = None
intrinsics_global = None


# ---------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    global depth_frame_global
    global intrinsics_global

    if event == cv2.EVENT_LBUTTONDOWN:

        if depth_frame_global is None:
            return

        # Get depth at clicked pixel in metres
        depth = depth_frame_global.get_distance(x, y)

        if depth <= 0:
            print("Invalid depth at this point.")
            return

        # Convert 2D pixel -> 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(
            intrinsics_global,
            [x, y],
            depth
        )

        clicked_points.append({
            "pixel": (x, y),
            "point": point_3d,
            "depth": depth
        })

        print("\nPoint selected:")
        print("Pixel:", x, y)
        print("Depth: {:.3f} mm".format(depth * 1000))
        print(
            "3D: X={:.3f} mm  Y={:.3f} mm  Z={:.3f} mm".format(
                point_3d[0] * 1000,
                point_3d[1] * 1000,
                point_3d[2] * 1000
            )
        )

        # Keep only the latest two points
        if len(clicked_points) > 2:
            clicked_points = clicked_points[-2:]

        if len(clicked_points) == 2:

            p1 = np.array(clicked_points[0]["point"])
            p2 = np.array(clicked_points[1]["point"])

            # True 3D Euclidean distance
            distance_m = np.linalg.norm(p2 - p1)

            distance_mm = distance_m * 1000

            print("\n===================================")
            print("Measured coin diameter:")
            print("{:.3f} mm".format(distance_mm))
            print("===================================\n")


# ---------------------------------------------------------
# Configure RealSense
# ---------------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()

# D405 depth stream
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

# Color stream
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

# Align depth to colour image
align = rs.align(rs.stream.color)

# ---------------------------------------------------------
# Camera warm-up
# ---------------------------------------------------------
print("Starting camera...")

for i in range(30):
    pipeline.wait_for_frames()

print("Camera ready.")
print()
print("INSTRUCTIONS")
print("---------------------------")
print("1. Put coin flat under camera.")
print("2. Click LEFT edge of coin.")
print("3. Click RIGHT edge of coin.")
print("4. Diameter will be calculated.")
print("5. Press R to reset.")
print("6. Press ESC to exit.")
print("---------------------------")


cv2.namedWindow("D405 Coin Measurement")
cv2.setMouseCallback(
    "D405 Coin Measurement",
    mouse_callback
)


try:

    while True:

        # -------------------------------------------------
        # Get frames
        # -------------------------------------------------
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # -------------------------------------------------
        # Get intrinsics
        # -------------------------------------------------
        depth_profile = depth_frame.profile.as_video_stream_profile()

        intrinsics_global = depth_profile.intrinsics
        depth_frame_global = depth_frame

        # -------------------------------------------------
        # Convert frames to numpy
        # -------------------------------------------------
        depth_image = np.asanyarray(
            depth_frame.get_data()
        )

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        # -------------------------------------------------
        # Make depth colour map
        # -------------------------------------------------
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(
                depth_image,
                alpha=0.03
            ),
            cv2.COLORMAP_JET
        )

        # Blend RGB + depth
        display = cv2.addWeighted(
            color_image,
            0.65,
            depth_colormap,
            0.35,
            0
        )

        # -------------------------------------------------
        # Draw selected points
        # -------------------------------------------------
        for i, item in enumerate(clicked_points):

            x, y = item["pixel"]

            cv2.circle(
                display,
                (x, y),
                6,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                display,
                "P{}".format(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # -------------------------------------------------
        # Draw line + measurement
        # -------------------------------------------------
        if len(clicked_points) == 2:

            p1_pixel = clicked_points[0]["pixel"]
            p2_pixel = clicked_points[1]["pixel"]

            cv2.line(
                display,
                p1_pixel,
                p2_pixel,
                (255, 255, 255),
                2
            )

            p1 = np.array(
                clicked_points[0]["point"]
            )

            p2 = np.array(
                clicked_points[1]["point"]
            )

            distance_mm = (
                np.linalg.norm(p2 - p1) * 1000
            )

            midpoint = (
                int((p1_pixel[0] + p2_pixel[0]) / 2),
                int((p1_pixel[1] + p2_pixel[1]) / 2)
            )

            text = "{:.2f} mm".format(distance_mm)

            cv2.putText(
                display,
                text,
                (midpoint[0] - 40, midpoint[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # -------------------------------------------------
        # Instructions overlay
        # -------------------------------------------------
        cv2.putText(
            display,
            "Click two opposite edges of coin",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            "R = Reset   ESC = Exit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "D405 Coin Measurement",
            display
        )

        key = cv2.waitKey(1)

        if key == 27:
            break

        elif key == ord("r") or key == ord("R"):
            clicked_points.clear()
            print("Measurement reset.")


finally:

    pipeline.stop()
    cv2.destroyAllWindows()
