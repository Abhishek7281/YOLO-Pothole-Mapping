# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# from utils.detect import detect_potholes  # Import the detect function

# st.set_page_config(page_title="YOLOv10 Pothole Detection", layout="wide")
# st.title("🛣️ YOLOv10 Pothole Detection System")

# uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

# if uploaded_file is not None:
#     is_video = uploaded_file.type.startswith("video/")

#     if is_video:
#         temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#         temp_video.write(uploaded_file.read())
#         temp_video_path = temp_video.name

#         video = cv2.VideoCapture(temp_video_path)
#         output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         fps = int(video.get(cv2.CAP_PROP_FPS))
#         frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             detected_frame = detect_potholes(frame)
#             out.write(detected_frame)

#         video.release()
#         out.release()
#         st.video(output_video_path)

#         with open(output_video_path, "rb") as file:
#             st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         os.remove(temp_video_path)
#         os.remove(output_video_path)

#     else:
#         image = Image.open(uploaded_file)
#         img_array = np.array(image)
#         detected_img = detect_potholes(img_array)
#         detected_pil = Image.fromarray(detected_img)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="Original Image", width=500)
#         with col2:
#             st.image(detected_pil, caption="Detected Potholes", width=500)

#         temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#         detected_pil.save(temp_image_path)

#         with open(temp_image_path, "rb") as file:
#             st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#         os.remove(temp_image_path)

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import torch
# from ultralytics import YOLO

# # Load YOLOv10n Model
# @st.cache_resource()  # Cache model for faster inference
# def load_model():
#     model_path = "project_files/best.pt"  # Updated to best.pt for YOLOv10n
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # Pothole Detection Function
# def detect_potholes(image):
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return image

# # Streamlit UI
# def main():
#     st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
#     st.title("🛣️ YOLOv10n Pothole Detection System")

#     uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             try:
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name

#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0

#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     detected_frame = detect_potholes(frame)
#                     out.write(detected_frame)
#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))

#                 video.release()
#                 out.release()
#                 st.success("✅ Video processing complete!")

#                 # Display Processed Video
#                 st.video(output_video_path)

#                 # Allow Download
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)

#             except Exception as e:
#                 st.error(f"❌ Error processing video: {e}")

#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=500)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=500)

#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)

#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st

# # ✅ `st.set_page_config()` must be the FIRST Streamlit command!
# st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")

# import asyncio
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import torch
# from ultralytics import YOLO

# # ✅ Fix RuntimeError: no running event loop (Torch issue)
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.run(asyncio.sleep(0))

# # Load YOLOv10n Model
# @st.cache_resource()  # Cache model for faster inference
# def load_model():
#     model_path = "project_files/best.pt"  # ✅ Use best.pt for YOLOv10n
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # Pothole Detection Function
# # def detect_potholes(image):
# #     results = model(image)

# #     for r in results:
# #         for box in r.boxes:
# #             x1, y1, x2, y2 = map(int, box.xyxy[0])
# #             confidence = float(box.conf[0])
# #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# #     return image

# def detect_potholes(image):
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             confidence = float(box.conf[0])  # Confidence score

#             # Draw thicker bounding box with deep color
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow box

#             # Create background for text
#             label = f"{confidence:.2f}"
#             (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
#             cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 255), -1)  # Filled bg

#             # Put text over the rectangle
#             cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)  # Black text

#     return image

# # Streamlit UI
# def main():
#     st.title("🛣️ YOLOv10n Pothole Detection System")
#     uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             try:
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name

#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0

#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     detected_frame = detect_potholes(frame)
#                     out.write(detected_frame)
#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))

#                 video.release()
#                 out.release()
#                 st.success("✅ Video processing complete!")

#                 # Display Processed Video
#                 st.video(output_video_path)

#                 # Allow Download
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)

#             except Exception as e:
#                 st.error(f"❌ Error processing video: {e}")

#         else:
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=500)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=500)

#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)

#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()

# import streamlit as st

# # ✅ `st.set_page_config()` must be the FIRST Streamlit command!
# st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")

# import asyncio
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import os
# import torch
# from ultralytics import YOLO

# # ✅ Fix RuntimeError: no running event loop (Torch issue)
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.run(asyncio.sleep(0))

# # ✅ Load YOLOv10n Model
# @st.cache_resource()  # Cache model for efficiency
# def load_model():
#     model_path = "project_files/best.pt"  # Path to YOLOv10n model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # ✅ Pothole Detection Function
# def detect_potholes(image):
#     """
#     Detect potholes using YOLOv10n and apply consistent bounding boxes and text.
#     """
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#             confidence = float(box.conf[0])  # Confidence score

#             # ✅ Define consistent styling
#             bbox_color = (0, 255, 0)  # ✅ Green Bounding Box
#             text_color = (0, 0, 0)  # Black Text
#             thickness = 4  # Bold bounding box
#             font_scale = 1.2
#             font_thickness = 3

#             # ✅ Draw Green bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

#             # ✅ Create background for text
#             label = f"{confidence:.2f}"
#             (text_width, text_height), _ = cv2.getTextSize(
#                 label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
#             )
#             cv2.rectangle(image, (x1, y1 - text_height - 10), 
#                           (x1 + text_width + 10, y1), bbox_color, -1)  # Filled bg

#             # ✅ Add text over the rectangle
#             cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         font_scale, text_color, font_thickness, cv2.LINE_AA)

#     return image

# # ✅ Streamlit UI
# def main():
#     st.title("🛣️ YOLOv10n Pothole Detection System")
#     uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             try:
#                 # ✅ Save uploaded video
#                 temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#                 temp_video.write(uploaded_file.read())
#                 temp_video_path = temp_video.name

#                 # ✅ Read video
#                 video = cv2.VideoCapture(temp_video_path)
#                 output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 fps = int(video.get(cv2.CAP_PROP_FPS))
#                 frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#                 # ✅ Process video frames
#                 progress_bar = st.progress(0)
#                 total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#                 frame_count = 0

#                 while True:
#                     ret, frame = video.read()
#                     if not ret:
#                         break
#                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                     detected_frame = detect_potholes(frame_rgb)
#                     detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV
#                     out.write(detected_frame)

#                     frame_count += 1
#                     progress_bar.progress(min(frame_count / total_frames, 1.0))

#                 # ✅ Release resources
#                 video.release()
#                 out.release()
#                 st.success("✅ Video processing complete!")

#                 # ✅ Display Processed Video
#                 st.video(output_video_path)

#                 # ✅ Allow Download
#                 with open(output_video_path, "rb") as file:
#                     st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#                 # ✅ Cleanup
#                 os.remove(temp_video_path)
#                 os.remove(output_video_path)

#             except Exception as e:
#                 st.error(f"❌ Error processing video: {e}")

#         else:
#             # ✅ Process image
#             image = Image.open(uploaded_file)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             # ✅ Display Original & Processed Images
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=625)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=625)

#             # ✅ Save & Download Processed Image
#             temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
#             detected_pil.save(temp_image_path)
#             with open(temp_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

#             # ✅ Cleanup
#             os.remove(temp_image_path)

# if __name__ == "__main__":
#     main()


# Original

# Increase file size
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import torch
# from ultralytics import YOLO

# # ✅ Increase File Upload Limit (Must be set in `.streamlit/config.toml` too)
# st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")

# # ✅ Load YOLOv10n Model
# @st.cache_resource()  # Cache model for faster processing

# def load_model():
#     model_path = "project_files/best.pt"  # ✅ Path to trained YOLOv10n model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# model = load_model()

# # ✅ Pothole Detection Function
# def detect_potholes(image):
#     """
#     Detect potholes using YOLOv10n and display bounding boxes & confidence scores.
#     """
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             confidence = float(box.conf[0])  # Confidence score

#             # ✅ Define styling
#             bbox_color = (0, 255, 0)  # Green bounding box
#             text_color = (0, 0, 0)  # Black text
#             thickness = 4  # Bold bounding box
#             font_scale = 1.2
#             font_thickness = 3

#             # ✅ Draw bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

#             # ✅ Create background for text
#             label = f"{confidence:.2f}"
#             (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(image, (x1, y1 - text_height - 10), 
#                           (x1 + text_width + 10, y1), bbox_color, -1)  # Filled bg

#             # ✅ Add confidence text
#             cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         font_scale, text_color, font_thickness, cv2.LINE_AA)

#     return image

# # ✅ Streamlit UI
# def main():
#     st.title("🛣️ YOLOv10n Pothole Detection System")

#     # ✅ File Uploader With Increased File Size Handling
#     uploaded_file = st.file_uploader("Upload an image or video (Up to 1GB)...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()  # ✅ Use Temp Directory
#         file_path = os.path.join(temp_dir, uploaded_file.name)

#         # ✅ Save large files to disk instead of RAM
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
#         st.success(f"✅ File uploaded: {uploaded_file.name} (Size: {file_size_mb} MB)")

#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             # ✅ Process video
#             video = cv2.VideoCapture(file_path)
#             output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#                 detected_frame = detect_potholes(frame_rgb)
#                 detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)  # Convert back for OpenCV
#                 out.write(detected_frame)

#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             # ✅ Release resources
#             video.release()
#             out.release()
#             st.success("✅ Video processing complete!")

#             # ✅ Display Processed Video
#             st.video(output_video_path)

#             # ✅ Allow Download
#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         else:
#             # ✅ Process image
#             image = Image.open(file_path)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array)
#             detected_pil = Image.fromarray(detected_img)

#             # ✅ Display Original & Processed Images
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=625)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=625)

#             # ✅ Save & Download Processed Image
#             output_image_path = os.path.join(temp_dir, "processed_image.png")
#             detected_pil.save(output_image_path)
#             with open(output_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

# if __name__ == "__main__":
#     main()

# original
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import torch
# from ultralytics import YOLO
# import asyncio

# def load_model():
#     model_path = "project_files/best.pt"  
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# def detect_potholes(image, model):
#     """
#     Detect potholes using YOLOv10n and display bounding boxes & confidence scores.
#     """
#     results = model(image)

#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  
#             confidence = float(box.conf[0])  

#             bbox_color = (0, 255, 0)  
#             text_color = (0, 0, 0)  
#             thickness = 4  
#             font_scale = 1.2
#             font_thickness = 3

#             cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

#             label = f"{confidence:.2f}"
#             (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#             cv2.rectangle(image, (x1, y1 - text_height - 10), 
#                           (x1 + text_width + 10, y1), bbox_color, -1)  

#             cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         font_scale, text_color, font_thickness, cv2.LINE_AA)

#     return image

# def main():
#     st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
#     st.title("🛣️ YOLOv10n Pothole Detection System")

#     try:
#         asyncio.get_running_loop()
#     except RuntimeError:
#         asyncio.run(asyncio.sleep(0))

#     model = load_model()  

#     uploaded_file = st.file_uploader("Upload an image or video (Up to 1GB)...", type=["jpg", "png", "jpeg", "mp4"])

#     if uploaded_file is not None:
#         temp_dir = tempfile.mkdtemp()  
#         file_path = os.path.join(temp_dir, uploaded_file.name)

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
#         st.success(f"✅ File uploaded: {uploaded_file.name} (Size: {file_size_mb} MB)")

#         is_video = uploaded_file.type.startswith("video/")

#         if is_video:
#             video = cv2.VideoCapture(file_path)
#             output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#             progress_bar = st.progress(0)
#             total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#             frame_count = 0

#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
#                 detected_frame = detect_potholes(frame_rgb, model)  # ✅ Pass model here
#                 detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_RGB2BGR)  
#                 out.write(detected_frame)

#                 frame_count += 1
#                 progress_bar.progress(min(frame_count / total_frames, 1.0))

#             video.release()
#             out.release()
#             st.success("✅ Video processing complete!")

#             st.video(output_video_path)

#             with open(output_video_path, "rb") as file:
#                 st.download_button("Download Processed Video", file, file_name="processed_video.mp4", mime="video/mp4")

#         else:
#             image = Image.open(file_path)
#             img_array = np.array(image)
#             detected_img = detect_potholes(img_array, model)  # ✅ Pass model here
#             detected_pil = Image.fromarray(detected_img)

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Original Image", width=625)
#             with col2:
#                 st.image(detected_pil, caption="Detected Potholes", width=625)

#             output_image_path = os.path.join(temp_dir, "processed_image.png")
#             detected_pil.save(output_image_path)
#             with open(output_image_path, "rb") as file:
#                 st.download_button("Download Processed Image", file, file_name="processed_image.png", mime="image/png")

# if __name__ == "__main__":
#     main()







# original 1
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import torch
# from ultralytics import YOLO
# import pandas as pd
# import zipfile

# def load_model():
#     model_path = "project_files/best.pt"  
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# def detect_potholes(image, model):
#     results = model(image)
#     pothole_data = []
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             pothole_data.append([x1, y1, x2, y2, confidence])
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#     return image, pothole_data

# def merge_gps_data(pothole_data, gps_data, frame_index):
#     merged_data = []
#     if frame_index < len(gps_data):
#         gps_lat, gps_lon = gps_data.iloc[frame_index][['Latitude', 'Longitude']]
#         for x1, y1, x2, y2, confidence in pothole_data:
#             merged_data.append([frame_index, gps_lat, gps_lon, x1, y1, x2, y2, confidence])
#     return merged_data

# def process_video(video_path, gps_data, model, temp_dir):
#     video = cv2.VideoCapture(video_path)
#     output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#     frames_folder = os.path.join(temp_dir, "frames")
#     os.makedirs(frames_folder, exist_ok=True)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     pothole_results = []
#     frame_index = 0
    
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break
#         detected_frame, pothole_data = detect_potholes(frame, model)
#         out.write(detected_frame)
#         frame_filename = f"frame_{frame_index:04d}.png"
#         cv2.imwrite(os.path.join(frames_folder, frame_filename), detected_frame)
#         pothole_results.extend(merge_gps_data(pothole_data, gps_data, frame_index))
#         frame_index += 1

#     video.release()
#     out.release()

#     pothole_df = pd.DataFrame(pothole_results, 
#         columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
#     pothole_csv_path = os.path.join(temp_dir, "pothole_coordinates.csv")
#     pothole_df.to_csv(pothole_csv_path, index=False)
    
#     return output_video_path, frames_folder, pothole_csv_path

# def process_image(image_path, model, temp_dir):
#     image = cv2.imread(image_path)
#     detected_img, _ = detect_potholes(image, model)
#     output_image_path = os.path.join(temp_dir, "processed_image.png")
#     cv2.imwrite(output_image_path, detected_img)
#     return output_image_path

# def main():
#     st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
#     st.title("🛣️ YOLOv10n Pothole Detection System with GPS")

#     model = load_model()

#     uploaded_file = st.file_uploader("Upload an image or video (Up to 1GB)...", type=["jpg", "jpeg", "png", "bmp", "tiff", "mp4", "avi", "mov"])
#     uploaded_gps = st.file_uploader("Upload GPS coordinates (CSV file)...", type=["csv"])

#     if uploaded_file and uploaded_gps:
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         gps_path = os.path.join(temp_dir, uploaded_gps.name)

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
#         with open(gps_path, "wb") as f:
#             f.write(uploaded_gps.read())

#         gps_data = pd.read_csv(gps_path)
#         is_video = uploaded_file.type.startswith("video/")
        
#         if is_video:
#             output_video_path, frames_folder, pothole_csv_path = process_video(file_path, gps_data, model, temp_dir)
#         else:
#             output_image_path = process_image(file_path, model, temp_dir)
#             frames_folder, pothole_csv_path = None, None
        
#         zip_path = os.path.join(temp_dir, "processed_results.zip")
#         with zipfile.ZipFile(zip_path, 'w') as zipf:
#             if is_video:
#                 zipf.write(output_video_path, "processed_video.mp4")
#                 zipf.write(pothole_csv_path, "pothole_coordinates.csv")
#                 for frame in os.listdir(frames_folder):
#                     zipf.write(os.path.join(frames_folder, frame), os.path.join("frames", frame))
#             else:
#                 zipf.write(output_image_path, "processed_image.png")
        
#         st.success("✅ Processing complete!")
        
#         if is_video:
#             st.video(output_video_path)
#         else:
#             st.image(output_image_path, caption="Detected Potholes", use_column_width=True)
        
#         with open(zip_path, "rb") as file:
#             st.download_button("Download All Processed Data (ZIP)", file, file_name="processed_results.zip", mime="application/zip")

# if __name__ == "__main__":
#     main()

# original 2
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
from ultralytics import YOLO
import pandas as pd
import zipfile

def load_model():
    model_path = "project_files/best.pt"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    return model

def detect_potholes(image, model):
    image_copy = image.copy()
    results = model(image_copy)
    pothole_data = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            if confidence > 0.5:  # Keep only high-confidence detections
                pothole_data.append([x1, y1, x2, y2, confidence])
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue box only
                cv2.putText(image_copy, f"{confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image_copy, pothole_data

def merge_gps_data(pothole_data, gps_data, frame_index):
    merged_data = []
    if not gps_data.empty and frame_index < len(gps_data):
        gps_lat, gps_lon = gps_data.iloc[frame_index][['Latitude', 'Longitude']]
        for x1, y1, x2, y2, confidence in pothole_data:
            merged_data.append([frame_index, gps_lat, gps_lon, x1, y1, x2, y2, confidence])
    return merged_data

def process_video(video_path, gps_data, model, temp_dir, progress_bar):
    video = cv2.VideoCapture(video_path)
    output_video_path = os.path.join(temp_dir, "processed_video.mp4")
    frames_folder = os.path.join(temp_dir, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    pothole_results = []
    frame_index = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        detected_frame, pothole_data = detect_potholes(frame, model)
        out.write(detected_frame)
        
        if pothole_data:
            frame_filename = f"frame_{frame_index:04d}.png"
            cv2.imwrite(os.path.join(frames_folder, frame_filename), detected_frame)
        
        pothole_results.extend(merge_gps_data(pothole_data, gps_data, frame_index))
        frame_index += 1
        
        # Update progress bar
        progress_bar.progress(frame_index / total_frames)
    
    video.release()
    out.release()
    
    pothole_df = pd.DataFrame(pothole_results, 
        columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
    pothole_csv_path = os.path.join(temp_dir, "pothole_coordinates.csv")
    pothole_df.to_csv(pothole_csv_path, index=False)
    
    return output_video_path, frames_folder, pothole_csv_path

def main():
    st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
    st.title("🛣️ YOLOv10n Pothole Detection System with GPS")
    
    if "model" not in st.session_state:
        st.session_state.model = load_model()
    
    uploaded_file = st.file_uploader("Upload a video (Up to 1TB)...", type=["mp4", "avi", "mov"])
    uploaded_gps = st.file_uploader("Upload GPS coordinates (CSV file)...", type=["csv"])
    
    if uploaded_file and uploaded_gps:
        if st.button("Start Processing"):
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            gps_path = os.path.join(temp_dir, uploaded_gps.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            with open(gps_path, "wb") as f:
                f.write(uploaded_gps.read())
            
            gps_data = pd.read_csv(gps_path)
            
            st.subheader("Processing video...")
            progress_bar = st.progress(0)
            
            output_video_path, frames_folder, pothole_csv_path = process_video(
                file_path, gps_data, st.session_state.model, temp_dir, progress_bar
            )
            
            zip_path = os.path.join(temp_dir, "processed_results.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(output_video_path, "processed_video.mp4")
                zipf.write(pothole_csv_path, "pothole_coordinates.csv")
                for frame in os.listdir(frames_folder):
                    zipf.write(os.path.join(frames_folder, frame), os.path.join("frames", frame))
            
            st.session_state.processed = {
                "output_video_path": output_video_path,
                "pothole_csv_path": pothole_csv_path,
                "zip_path": zip_path
            }
            st.session_state.download_clicked = False  # Track download state
    
    if "processed" in st.session_state and not st.session_state.get("download_clicked", False):
        st.success("✅ Processing complete!")
        st.video(st.session_state.processed["output_video_path"])
        
        with open(st.session_state.processed["zip_path"], "rb") as file:
            if st.download_button("Download All Processed Data (ZIP)", file, file_name="processed_results.zip", mime="application/zip"):
                st.session_state.download_clicked = True  # Mark download as clicked
                st.session_state.clear()  # Clear session state
                st.rerun()  # Refresh the app

if __name__ == "__main__":
    main()
    
# import streamlit as st
# import os
# import cv2
# import numpy as np
# import tempfile
# import zipfile
# import pandas as pd

# # ✅ Increase Upload Limit to 1GB
# os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"

# # ✅ Load YOLOv4-tiny Model
# def load_model():
#     net = cv2.dnn.readNet("project_files/yolov4_tiny.weights", "project_files/yolov4_tiny.cfg")
#     conf_threshold = 0.25
#     nms_threshold = 0.4
#     model = cv2.dnn_DetectionModel(net)
#     model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
#     return model, conf_threshold, nms_threshold

# # ✅ Detection Function
# def detect_potholes(img, model, conf_threshold, nms_threshold):
#     class_ids, confidences, boxes = model.detect(img, confThreshold=conf_threshold, nmsThreshold=nms_threshold)

#     if len(class_ids) == 0:
#         return img, []

#     detected_boxes = []
#     for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
#         x, y, w, h = map(int, box)
#         pothole_class_id = 0  # Adjust if potholes use a different ID
#         if class_id == pothole_class_id:
#             detected_boxes.append((x, y, x + w, y + h, float(confidence)))
#             color = (139, 0, 0)
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
#             cv2.putText(img, f"Pothole {confidence:.2f}", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     return img, detected_boxes

# # ✅ Streamlit App
# def main():
#     st.set_page_config(page_title="Pothole Detection", layout="wide")
#     st.title("🛣️ Pothole Detection System")

#     if "model" not in st.session_state:
#         st.session_state.model, st.session_state.conf_threshold, st.session_state.nms_threshold = load_model()

#     uploaded_media = st.file_uploader(
#         "Upload Image or Video (MP4, AVI, JPG, PNG, etc.)", 
#         type=["mp4", "avi", "mov", "mkv", "flv", "webm", "jpg", "jpeg", "png", "bmp", "tiff"]
#     )

#     uploaded_gps = st.file_uploader("Upload GPS Coordinates CSV (Mandatory)", type=["csv"])

#     process_button = st.button("Start Processing")

#     if process_button and uploaded_media and uploaded_gps:
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, uploaded_media.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_media.read())

#         gps_df = pd.read_csv(uploaded_gps)
#         is_image = uploaded_media.type.startswith("image")

#         detection_data = []
#         frames_dir = os.path.join(temp_dir, "frames")
#         os.makedirs(frames_dir, exist_ok=True)

#         if is_image:
#             frame = cv2.imread(file_path)
#             detected_frame, boxes = detect_potholes(
#                 frame, st.session_state.model,
#                 st.session_state.conf_threshold,
#                 st.session_state.nms_threshold
#             )

#             frame_path = os.path.join(frames_dir, "frame_0000.png")
#             cv2.imwrite(frame_path, detected_frame)

#             latitude, longitude = (gps_df.iloc[0]['Latitude'], gps_df.iloc[0]['Longitude']) if not gps_df.empty else (None, None)
#             for (x1, y1, x2, y2, confidence) in boxes:
#                 detection_data.append(["frame_0000.png", x1, y1, x2, y2, confidence, latitude, longitude])

#             output_video_path = frame_path  # For consistency
#         else:
#             video = cv2.VideoCapture(file_path)
#             fps = int(video.get(cv2.CAP_PROP_FPS))
#             width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out_path = os.path.join(temp_dir, "processed_video.mp4")
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#             frame_index = 0
#             while True:
#                 ret, frame = video.read()
#                 if not ret:
#                     break

#                 detected_frame, boxes = detect_potholes(
#                     frame, st.session_state.model,
#                     st.session_state.conf_threshold,
#                     st.session_state.nms_threshold
#                 )

#                 frame_filename = f"frame_{frame_index:04d}.png"
#                 cv2.imwrite(os.path.join(frames_dir, frame_filename), detected_frame)

#                 latitude, longitude = (None, None)
#                 if frame_index < len(gps_df):
#                     latitude, longitude = gps_df.iloc[frame_index][['Latitude', 'Longitude']]

#                 for (x1, y1, x2, y2, confidence) in boxes:
#                     detection_data.append([frame_filename, x1, y1, x2, y2, confidence, latitude, longitude])

#                 out.write(detected_frame)
#                 frame_index += 1

#             video.release()
#             out.release()
#             output_video_path = out_path

#         # ✅ Save Detection Data
#         excel_path = os.path.join(temp_dir, "pothole_coordinates.xlsx")
#         pd.DataFrame(detection_data, columns=[
#             "Frame", "X1", "Y1", "X2", "Y2", "Confidence", "Latitude", "Longitude"
#         ]).to_excel(excel_path, index=False)

#         # ✅ ZIP All Results
#         zip_path = os.path.join(temp_dir, "processed_results.zip")
#         with zipfile.ZipFile(zip_path, 'w') as zipf:
#             zipf.write(output_video_path, os.path.basename(output_video_path))
#             zipf.write(excel_path, "pothole_coordinates.xlsx")
#             for frame in os.listdir(frames_dir):
#                 zipf.write(os.path.join(frames_dir, frame), os.path.join("frames", frame))

#         # ✅ Show Download Button after processing
#         st.success("✅ Processing complete!")
#         with open(zip_path, "rb") as f:
#             st.download_button("📦 Download All Processed Data (ZIP)", f,
#                                file_name="processed_results.zip", mime="application/zip")

#         # ✅ Optional: Manual reset button
#         if st.button("🔁 Reset App"):
#             st.session_state.clear()
#             st.rerun()

# if __name__ == "__main__":
#     main()




# Data process not in ram
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile
# import torch
# from ultralytics import YOLO
# import pandas as pd
# import zipfile

# def load_model():
#     model_path = "project_files/best.pt"  
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = YOLO(model_path).to(device)
#     return model

# def detect_potholes(image, model):
#     results = model(image)
#     pothole_data = []
#     for r in results:
#         for box in r.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             confidence = float(box.conf[0])
#             pothole_data.append([x1, y1, x2, y2, confidence])
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.putText(image, f"{confidence:.2f}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
#     return image, pothole_data

# def merge_gps_data(pothole_data, gps_data, frame_index):
#     merged_data = []
#     if frame_index < len(gps_data):
#         gps_lat, gps_lon = gps_data.iloc[frame_index][['Latitude', 'Longitude']]
#         for x1, y1, x2, y2, confidence in pothole_data:
#             merged_data.append([frame_index, gps_lat, gps_lon, x1, y1, x2, y2, confidence])
#     return merged_data

# def process_video(video_path, gps_data, model, temp_dir):
#     video = cv2.VideoCapture(video_path)
#     output_video_path = os.path.join(temp_dir, "processed_video.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     fps = int(video.get(cv2.CAP_PROP_FPS))
#     frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

#     pothole_results = []
#     frame_index = 0
#     with st.status("Processing video... Please wait ⏳", expanded=True) as status:
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             detected_frame, pothole_data = detect_potholes(frame, model)
#             out.write(detected_frame)
#             pothole_results.extend(merge_gps_data(pothole_data, gps_data, frame_index))
#             frame_index += 1
        
#         video.release()
#         out.release()
        
#         pothole_df = pd.DataFrame(pothole_results, 
#             columns=["Frame", "Latitude", "Longitude", "X1", "Y1", "X2", "Y2", "Confidence"])
#         pothole_csv_path = os.path.join(temp_dir, "pothole_coordinates.csv")
#         pothole_df.to_csv(pothole_csv_path, index=False)
        
#         status.update(label="✅ Processing complete!", state="complete")
    
#     return output_video_path, pothole_csv_path

# def process_image(image_path, model, temp_dir):
#     image = cv2.imread(image_path)
#     detected_img, _ = detect_potholes(image, model)
#     output_image_path = os.path.join(temp_dir, "processed_image.png")
#     cv2.imwrite(output_image_path, detected_img)
#     return output_image_path

# def main():
#     st.set_page_config(page_title="YOLOv10n Pothole Detection", layout="wide")
#     st.title("🛣️ YOLOv10n Pothole Detection System with GPS")

#     model = load_model()

#     uploaded_file = st.file_uploader("Upload an image or video (Up to 1GB)...", type=["jpg", "jpeg", "png", "bmp", "tiff", "mp4", "avi", "mov"])
#     uploaded_gps = st.file_uploader("Upload GPS coordinates (CSV file)...", type=["csv"])

#     if uploaded_file and uploaded_gps:
#         temp_dir = tempfile.mkdtemp()
#         file_path = os.path.join(temp_dir, uploaded_file.name)
#         gps_path = os.path.join(temp_dir, uploaded_gps.name)

#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
#         with open(gps_path, "wb") as f:
#             f.write(uploaded_gps.read())

#         gps_data = pd.read_csv(gps_path)
#         is_video = uploaded_file.type.startswith("video/")
        
#         if is_video:
#             output_video_path, pothole_csv_path = process_video(file_path, gps_data, model, temp_dir)
#         else:
#             output_image_path = process_image(file_path, model, temp_dir)
#             pothole_csv_path = None
        
#         zip_path = os.path.join(temp_dir, "processed_results.zip")
#         with zipfile.ZipFile(zip_path, 'w') as zipf:
#             if is_video:
#                 zipf.write(output_video_path, "processed_video.mp4")
#                 zipf.write(pothole_csv_path, "pothole_coordinates.csv")
#             else:
#                 zipf.write(output_image_path, "processed_image.png")
        
#         st.success("✅ Processing complete!")
        
#         if is_video:
#             st.video(output_video_path)
#         else:
#             st.image(output_image_path, caption="Detected Potholes", use_column_width=True)
        
#         with open(zip_path, "rb") as file:
#             st.download_button("Download Processed Data (ZIP)", file, file_name="processed_results.zip", mime="application/zip")

# if __name__ == "__main__":
#     main()
