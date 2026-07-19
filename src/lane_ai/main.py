from pathlib import Path
import sys
import argparse

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lane_ai.video import VideoIO
from lane_ai.lane_detector import BikeCorridorDetector
from lane_ai.overlay import draw_bike_corridor


def main():
    parser = argparse.ArgumentParser(description="Bike corridor detector")
    parser.add_argument("--input", default="data/input/input.mp4")
    parser.add_argument("--output", default="outputs/bike_corridor_output.mp4")
    args = parser.parse_args()

    print("Bike corridor detector starting...")

    video = VideoIO(args.input, args.output)
    detector = BikeCorridorDetector()

    while True:
        success, frame = video.read()
        if not success:
            break

        result = detector.detect(frame)
        final_frame = draw_bike_corridor(frame, result)

        cv2.imshow("Bike Corridor", final_frame)
        video.write(final_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()