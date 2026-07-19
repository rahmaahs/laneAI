from pathlib import Path
import sys
import argparse

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lane_ai.video import VideoIO
from lane_ai.lane_detector import LaneDetector
from lane_ai.overlay import display_lines, draw_status


print("Lane AI starting...")


def main():
    parser = argparse.ArgumentParser(description="Lane AI")
    parser.add_argument(
        "--input",
        default="data/input/input.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="outputs/lane_output.mp4",
        help="Path to output video",
    )
    args = parser.parse_args()

    video = VideoIO(args.input, args.output)
    detector = LaneDetector()

    while True:
        success, frame = video.read()
        if not success:
            break

        results = detector.detect(frame)
        lane_lines = results["lane_lines"]

        line_image = display_lines(frame, lane_lines)
        combo_image = cv2.addWeighted(frame, 0.9, line_image, 1.0, 1.0)
        final_image = draw_status(combo_image, lane_lines)

        cv2.imshow("Lane AI", final_image)
        video.write(final_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()