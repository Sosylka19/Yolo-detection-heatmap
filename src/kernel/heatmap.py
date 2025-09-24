import cv2
import numpy as np
from typing import Iterator
from dataclasses import dataclass
import uuid

from src.kernel.model import Frame


SUBSAMPLE = 2

@dataclass
class HeatmapFrame(Frame):
    overlay_bgr: np.ndarray

class Heatmap:
    def __init__(self, video, color, start: int, end: int):
        self.cap = cv2.VideoCapture(video)

        self.video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        self.colormap = color
        self.start_heatmap = start
        self.end_heatmap = end


    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Return preprocessed image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 25)
        img_canny = cv2.Canny(img_blur, 5, 50)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
        img_erode = cv2.erode(img_dilate, kernel, iterations=1)
        return img_erode
    
    def frames(self) -> Iterator[HeatmapFrame]:
        ok, img1 = self.cap.read()
        if not ok:
            return
        ok, img2 = self.cap.read()
        if not ok:
            return
        
        heat_map = np.zeros(img1.shape[:2], dtype=np.float32)

        while ok:
            diff = cv2.absdiff(img1, img2)
            mask = self.process(diff)

            
            cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            img_contours = np.zeros_like(img1)
            cv2.drawContours(img_contours, cnts, -1, (0, 255, 0), -1)

            green = np.all(img_contours == [0, 255, 0], axis = 2)
            heat_map[green] += 3
            heat_map[~green] -= 3
            np.clip(heat_map, 0, 255, out=heat_map)

            overlay = cv2.applyColorMap(heat_map.astype(np.uint8), self.colormap)

            yield HeatmapFrame(frame_bgr=img1, mask=mask, overlay_bgr=overlay)

            img1 = img2
            ok, img2 = self.cap.read()

        self.cap.release()

    def output(self):
        """
        Генератором возвращаю кадры для стриминга
        """
        width = max(1, int(self.src_w * 0.5))
        height = max(1, int(self.src_h * 0.5))
        desired_fps = self.fps // SUBSAMPLE
        frame_per_sec = max(1, int(desired_fps * 2.0))

        writer = None
        written_in_segment = 0
        written_in_segment_all = 0

        i = 0
        output_video_name = None

        try:
            for f in self.frames():
                start_frame = f.frame_bgr
                frame = f.overlay_bgr
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    start_frame = cv2.resize(start_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                if writer is None:
                    output_video_name = f"data/heatmap_{uuid.uuid4()}.mp4"
                    writer = cv2.VideoWriter(output_video_name, self.video_codec, 
                                             float(desired_fps), (width, height))
                    written_in_segment = 0

                if (i % SUBSAMPLE) == 0:
                    if written_in_segment_all >= self.start_heatmap * desired_fps and \
                    written_in_segment_all <= self.end_heatmap * desired_fps:
                        writer.write(frame)
                    else:
                        writer.write(start_frame)
                    written_in_segment += 1
                    written_in_segment_all += 1

                i += 1

                if written_in_segment >= frame_per_sec:
                    writer.release()
                    writer = None
                    yield output_video_name
                    output_video_name = None

            if writer is not None and written_in_segment > 0:
                writer.release()
                yield output_video_name

        finally:
            if writer is not None:
                writer.release()


# import cv2
# import os

# if __name__ == "__main__":
#     test_video = "data/5497_Francisco_San_1280x720.mp4"  
#     h = Heatmap(test_video)
#     first = next(h.output())
#     cap = cv2.VideoCapture(first)
#     for frame in h.output():
#         cap = cv2.VideoCapture(frame)
#         ok, fr = cap.read()
#         cv2.imshow("Bla",fr)
#     cap.release()

    


#     cv2.destroyAllWindows()
            