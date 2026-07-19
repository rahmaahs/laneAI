from typing import Optional

import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        self.prev_left_fit = None
        self.prev_right_fit = None

    def canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        polygons = np.array([[
            (int(0.10 * width), height),
            (int(0.90 * width), height),
            (int(0.60 * width), int(0.55 * height)),
            (int(0.40 * width), int(0.55 * height)),
        ]])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def make_coordinates(self, image, line_parameters):
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)

        if abs(slope) < 1e-6:
            slope = 1e-6

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def average_slope_intercept(self, image, lines):
        left_fit = []
        right_fit = []

        if lines is None:
            return None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            if x1 == x2:
                continue

            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_line = None
        right_line = None

        if len(left_fit) > 0:
            left_fit_average = np.average(left_fit, axis=0)
            self.prev_left_fit = left_fit_average
            left_line = self.make_coordinates(image, left_fit_average)
        elif self.prev_left_fit is not None:
            left_line = self.make_coordinates(image, self.prev_left_fit)

        if len(right_fit) > 0:
            right_fit_average = np.average(right_fit, axis=0)
            self.prev_right_fit = right_fit_average
            right_line = self.make_coordinates(image, right_fit_average)
        elif self.prev_right_fit is not None:
            right_line = self.make_coordinates(image, self.prev_right_fit)

        lines_out = []
        if left_line is not None:
            lines_out.append(left_line)
        if right_line is not None:
            lines_out.append(right_line)

        if len(lines_out) == 0:
            return None

        return np.array(lines_out)

    def detect(self, frame):
        edges = self.canny(frame)
        roi = self.region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=2,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=40,
            maxLineGap=5,
        )

        averaged_lines = self.average_slope_intercept(frame, lines)

        return {
            "edges": edges,
            "roi": roi,
            "lane_lines": averaged_lines,
        }