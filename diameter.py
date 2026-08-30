import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "D405 Vision Measurement"
MENU_NAME = "Select Measurement Type"
OBJECT_OPTIONS = ["Coin", "Ring", "Bar", "Chain"]

clicked_points = []
depth_frame_global = None
intrinsics_global = None
selected_object = None


def draw_menu():
    """Show the object selector without starting the camera."""
    menu = np.full((420, 640, 3), (32, 35, 42), dtype=np.uint8)
    cv2.putText(menu, "Select an object", (170, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(menu, "The camera starts after your selection", (130, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 190, 200), 1, cv2.LINE_AA)

    button_width, button_height = 220, 85
    positions = [(75, 145), (345, 145), (75, 265), (345, 265)]
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
    global clicked_points
    if event != cv2.EVENT_LBUTTONDOWN or depth_frame_global is None:
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
        print("\n===================================")
        print("Measured {} size:".format(selected_object.lower()))
        print("{:.3f} mm".format(distance_mm))
        print("===================================\n")


def draw_measurement(display):
    for index, item in enumerate(clicked_points):
        x, y = item["pixel"]
        cv2.circle(display, (x, y), 6, (255, 255, 255), -1)
        cv2.putText(display, "P{}".format(index + 1), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)

    if len(clicked_points) != 2:
        return

    p1_pixel = clicked_points[0]["pixel"]
    p2_pixel = clicked_points[1]["pixel"]
    cv2.line(display, p1_pixel, p2_pixel, (255, 255, 255), 2)
    p1 = np.array(clicked_points[0]["point"])
    p2 = np.array(clicked_points[1]["point"])
    distance_mm = np.linalg.norm(p2 - p1) * 1000
    midpoint = (int((p1_pixel[0] + p2_pixel[0]) / 2),
                int((p1_pixel[1] + p2_pixel[1]) / 2))
    cv2.putText(display, "{:.2f} mm".format(distance_mm),
                (midpoint[0] - 40, midpoint[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


def run_camera():
    global depth_frame_global, intrinsics_global
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # OpenCV expects BGR, so request that channel order directly from RealSense.
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print("Starting camera for {} measurement...".format(selected_object.lower()))
    for _ in range(30):
        pipeline.wait_for_frames()
    print("Camera ready.")
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
            display = np.asanyarray(color_frame.get_data()).copy()
            draw_measurement(display)
            cv2.putText(display,
                        "Mode: {} | Click two measurement points".format(selected_object),
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, "R = Reset   M = Menu   ESC = Exit", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False
            if key in (ord("r"), ord("R")):
                clicked_points.clear()
                print("Measurement reset.")
            if key in (ord("m"), ord("M")):
                return True
    finally:
        depth_frame_global = None
        intrinsics_global = None
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
