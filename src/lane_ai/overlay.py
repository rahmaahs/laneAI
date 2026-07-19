import cv2
import numpy as np


def draw_bike_corridor(frame, result):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    left_line = result["left_line"]
    right_line = result["right_line"]
    center_x = result["center_x"]
    offset_px = result["offset_px"]
    status = result["status"]

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 10)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 10)

    if center_x is not None:
        cv2.line(overlay, (center_x, h), (center_x, int(h * 0.6)), (255, 255, 255), 2)

    frame_center_x = w // 2
    cv2.line(overlay, (frame_center_x, h), (frame_center_x, int(h * 0.6)), (0, 0, 255), 2)

    cv2.rectangle(overlay, (20, 20), (410, 135), (0, 0, 0), -1)
    cv2.putText(overlay, "BIKE CORRIDOR", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if offset_px is None:
        offset_text = "OFFSET: --"
    else:
        offset_text = f"OFFSET: {offset_px:+d}px"

    cv2.putText(overlay, offset_text, (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"STATUS: {status}", (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    return overlay