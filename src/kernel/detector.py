## ЗДЕСЬ ТЯЖЕЛЫЙ ИНФЕРЕНС НА RTDetr



# import cv2
# import numpy as np
# from typing import Iterator
# import uuid
# from huggingface_hub import snapshot_download
# import torch
# from PIL import Image

# from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
# from src.kernel.draw_boxes import draw_bounding_boxes

# CUDA = torch.cuda.is_available()

# local_dir = snapshot_download("PekingU/rtdetr_r50vd")
# image_processor = RTDetrImageProcessor.from_pretrained(local_dir)
# if CUDA:
#     model = RTDetrForObjectDetection.from_pretrained(local_dir).to("cuda")
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     model = RTDetrForObjectDetection.from_pretrained(local_dir).to("mps")
# else:
#     model = RTDetrForObjectDetection.from_pretrained(local_dir).to("cpu")


# SUBSAMPLE = 4

# class Detector:
#     def __init__(self, video, conf_threshold):
#         self.cap = cv2.VideoCapture(video)
#         self.threshold = conf_threshold
#         self.video_codec = cv2.VideoWriter_fourcc(*"mp4v")
#         self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         self.desired_fps = self.fps // SUBSAMPLE
#         self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
#         self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

#     def stream_object_detection(self):

#         ok, frame = self.cap.read()

#         n_frames = 0

#         output_video_name = f"data/output_{uuid.uuid4()}.mp4"

#         output_video = cv2.VideoWriter(output_video_name, self.video_codec, self.desired_fps, 
#                                     (self.width, self.height))
#         batch = []

#         while ok:
#             frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             if n_frames % SUBSAMPLE == 0:
#                 batch.append(frame)
#             if len(batch) == 2 * self.desired_fps:
#                 if CUDA:
#                     inputs = image_processor(images=batch, return_tensors="pt").to("cuda")
#                 elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#                     inputs = image_processor(images=batch, return_tensors="pt").to("mps")
#                 else:
#                     inputs = image_processor(images=batch, return_tensors="pt").to("cpu")

#                 with torch.no_grad():
#                     outputs = model(**inputs)

                
#                 boxes = image_processor.post_process_object_detection(
#                     outputs, 
#                     target_sizes=torch.tensor([(self.height, self.width)] * len(batch)),
#                     threshold=self.threshold
#                 )

#                 for i, (array, box) in enumerate(zip(batch, boxes)):
#                     pil_image = draw_bounding_boxes(Image.fromarray(array), box, 
#                                                     model, self.threshold)
#                     frame = np.array(pil_image)
#                     frame = frame[:, :, ::-1].copy()
#                     output_video.write(frame)

#                 batch = []
#                 output_video.release()
#                 yield output_video_name
#                 output_video_name = f"data/output_{uuid.uuid4()}.mp4"
#                 output_video = cv2.VideoWriter(output_video_name, self.video_codec,
#                                             self.desired_fps, (self.width, self.height))
                
#             ok, frame = self.cap.read()
#             n_frames += 1

#тут проверка работы на локалке 


# import cv2
# import os

# if __name__ == "__main__":
#     test_video = "/Users/aleksandrandreev/cg/cg/data/5497_Francisco_San_1280x720.mp4"   # путь до видео для проверки
#     if not os.path.exists(test_video):
#         raise FileNotFoundError(f"Нет файла {test_video}")

#     for out_path in stream_object_detection(test_video, conf_threshold=0.3):
#         cap = cv2.VideoCapture(out_path)
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             cv2.imshow("Detector output", frame)
#             if cv2.waitKey(30) & 0xFF == 27:
#                 break
#         cap.release()

#     cv2.destroyAllWindows()
            

# class Detector:
#     def __init__(self, path: Path, min_contour_area: int):
#         self.cap = cv2.VideoCapture(str(path))
#         self.backSub = cv2.createBackgroundSubtractorMOG2(
#             history=1000,
#             varThreshold=30,
#             detectShadows=False
#         )
#         self.min_contour_area = min_contour_area
#         if not self.cap.isOpened():
#             return ValueError(f"Cannot open video: {path}")
        
#     def detect(self) -> Iterator[Frame]:
#         ok, img = self.cap.read()
#         if not ok:
#             return

#         while ok:
#             fg_mask = self.backSub.apply(img, learningRate=0.006)
            
#             _, mask_threshold = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#             mask_eroded = cv2.morphologyEx(mask_threshold, cv2.MORPH_OPEN, kernel)
#             contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]

#             frame_out = img.copy()
#             for cnt in large_contours:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

#             yield Frame(frame_bgr=frame_out, mask=mask_eroded)

#             ok, img = self.cap.read()

#         self.cap.release()

                   
    

    
