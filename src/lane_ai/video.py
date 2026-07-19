from pathlib import Path

import cv2


class VideoIO:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

        input_file = Path(input_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(input_file))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {input_file}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_file}")

    def read(self):
        return self.cap.read()

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()