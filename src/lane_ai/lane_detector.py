from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class CorridorResult:
    left_line: np.ndarray | None
    right_line: np.ndarray | None
    center_x: int | None
    offset_px: int | None
    status: str


class BikeCorridorDetector:
    def __init__(self, smooth_alpha: float = 0.8):
        self.smooth_alpha = smooth_alpha
        self.prev_left_line = None
        self.prev_right_line = None

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, image):
        height, width = image.shape[:2]

        # Tuned for a bike-lane / curb corridor view.
        polygons = np.array([[
            (int(0.10 * width), height),
            (int(0.98 * width), height),
            (int(0.72 * width), int(0.58 * height)),
            (int(0.40 * width), int(0.58 * height)),
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        return cv2.bitwise_and(image, mask)

    def _line_length(self, line):
        x1, y1, x2, y2 = line
        return math.hypot(x2 - x1, y2 - y1)

    def _x_at_y(self, line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return None
        m = (x2 - x1) / (y2 - y1)
        b = x1 - m * y1
        return m * y + b

    def _fit_line(self, segments, height):
        if not segments:
            return None

        xs = []
        ys = []
        weights = []

        for seg in segments:
            x1, y1, x2, y2 = seg
            length = self._line_length(seg)
            xs.extend([x1, x2])
            ys.extend([y1, y2])
            weights.extend([length, length])

        if len(xs) < 2:
            return None

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)

        # Fit x as a function of y: x = m*y + b
        m, b = np.polyfit(ys, xs, 1, w=weights)

        y1 = height
        y2 = int(height * 0.60)
        x1 = int(m * y1 + b)
        x2 = int(m * y2 + b)

        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def _smooth(self, current, previous):
        if current is None:
            return previous
        if previous is None:
            return current.astype(np.float32)
        return self.smooth_alpha * previous + (1.0 - self.smooth_alpha) * current

    def detect(self, frame):
        height, width = frame.shape[:2]

        edges = self.canny(frame)
        roi = self.region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=2,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=40,
            maxLineGap=60,
        )

        left_segments = []
        right_segments = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)

                length = self._line_length((x1, y1, x2, y2))
                if length < 35:
                    continue

                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 15 or angle > 85:
                    continue

                mid_x = (x1 + x2) / 2.0
                bottom_x = self._x_at_y((x1, y1, x2, y2), height - 1)
                if bottom_x is None:
                    continue

                # Left painted line should sit left of the corridor center.
                if mid_x < width * 0.58 and width * 0.08 <= bottom_x <= width * 0.70:
                    left_segments.append((x1, y1, x2, y2))

                # Right boundary is the curb edge / sidewalk edge.
                if mid_x >= width * 0.42 and width * 0.35 <= bottom_x <= width * 0.98:
                    right_segments.append((x1, y1, x2, y2))

        left_line = self._fit_line(left_segments, height)
        right_line = self._fit_line(right_segments, height)

        left_line = self._smooth(left_line, self.prev_left_line)
        right_line = self._smooth(right_line, self.prev_right_line)

        self.prev_left_line = left_line
        self.prev_right_line = right_line

        left_bottom = None
        right_bottom = None

        if left_line is not None:
            left_bottom = int(left_line[0])
        if right_line is not None:
            right_bottom = int(right_line[0])

        status = "SEARCHING"
        center_x = None
        offset_px = None

        if left_bottom is not None and right_bottom is not None:
            center_x = int((left_bottom + right_bottom) / 2)
            frame_center_x = width // 2
            offset_px = center_x - frame_center_x

            if abs(offset_px) < 40:
                status = "CENTER"
            elif offset_px < 0:
                status = "SHIFT LEFT"
            else:
                status = "SHIFT RIGHT"

        return {
            "edges": edges,
            "roi": roi,
            "left_line": None if left_line is None else np.round(left_line).astype(np.int32),
            "right_line": None if right_line is None else np.round(right_line).astype(np.int32),
            "center_x": center_x,
            "offset_px": offset_px,
            "status": status,
        }