import cv2
import gradio as gr

from src.gui.interface import get_generators



with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            conf_threshhold = gr.Slider(
                    label = "Confidence threshold",
                    minimum = 0.0,
                    maximum = 1.0,
                    step = 0.05,
                    value = 0.30
                )
                #доделать тут
        with gr.Column():
            colormap = gr.Radio(
                    ["Autumn", "Bone", "Jet", "HSV", "Inferno"],
                     label="Choose Colormap(OpenCV)",
                    value="HSV"
                )
        with gr.Column():
             timezone_start = gr.Number(
                 label="Write start of video(seconds)"
                 )

        with gr.Column():
            timezone_end = gr.Number(
                 label="Write end of video(seconds)"
                 )
            

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label = "Video Source")
    with gr.Row():
        submit = gr.Button("Start")
    with gr.Row():
        with gr.Column():
            output_video_detector = gr.Video(label = "Detected video",  streaming=True,autoplay=True)
        with gr.Column():
            output_video_heatmap = gr.Video(label = "Heatmap",streaming=True, autoplay=True)


    video_input.upload(
                fn = get_generators,
                inputs = [video_input, conf_threshhold, colormap, timezone_start, timezone_end],
                outputs = [ output_video_heatmap, output_video_detector]
    )

    demo.queue()


if __name__ == "__main__":
    demo.launch()