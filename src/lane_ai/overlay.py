import cv2
import numpy as np


def display_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    return line_image


def draw_status(frame, lane_lines):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    status_text = "NO LANES"
    steering_text = "UNKNOWN"

    if lane_lines is not None and len(lane_lines) >= 2:
        left_line = lane_lines[0]
        right_line = lane_lines[1]

        left_bottom_x = int(left_line[0])
        right_bottom_x = int(right_line[0])

        lane_center_x = int((left_bottom_x + right_bottom_x) / 2)
        frame_center_x = int(w / 2)
        offset = lane_center_x - frame_center_x

        if abs(offset) < 40:
            steering_text = "CENTER"
        elif offset < 0:
            steering_text = "MOVE LEFT"
        else:
            steering_text = "MOVE RIGHT"

        status_text = f"OFFSET: {offset}px"

        cv2.line(
            overlay,
            (frame_center_x, h),
            (frame_center_x, int(h * 0.6)),
            (255, 255, 255),
            2,
        )
        cv2.line(
            overlay,
            (lane_center_x, h),
            (lane_center_x, int(h * 0.6)),
            (0, 255, 255),
            2,
        )

    cv2.rectangle(overlay, (20, 20), (350, 125), (0, 0, 0), -1)
    cv2.putText(overlay, "LANE AI", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, status_text, (35, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, steering_text, (35, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return overlay