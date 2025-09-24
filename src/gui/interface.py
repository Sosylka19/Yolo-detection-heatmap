import gradio as gr
import cv2
from src.kernel.detector import Detector
from src.kernel.heatmap import Heatmap

colors = {
    "Autumn": cv2.COLORMAP_AUTUMN,
    "Bone": cv2.COLORMAP_BONE,
    "Jet": cv2.COLORMAP_JET,
    "HSV": cv2.COLORMAP_HSV,
    "Inferno": cv2.COLORMAP_INFERNO
}

def get_generators(video, threshold, colormap: str, start: str, end: str):
    color = colors.get(colormap)
    video_duration = gr.Video.get_video_duration_ffprobe(video)

    d = Detector(video, threshold)
    if int(start) < 0.0 or int(end) >= video_duration:
        raise gr.Error("Write valid interval", duration=5)
    
    h = Heatmap(video, color, int(start), int(end))
    for path_heatmap in h.output():
        yield (
            path_heatmap,
            gr.skip()
        )
    for path_det in d.stream_object_detection():
        yield (
            gr.skip(),
            path_det
        )
