# Pothole Detection using YOLOv10n (and YOLO11n/YOLO11s)

This repository demonstrates pothole detection in images and video using the [YOLOv10n](https://github.com/WongKinYiu/yolov10) model. The process is also compatible with YOLO11n and YOLO11s—simply swap the weights as described below to test or deploy different versions.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Process](#training-process)
- [Inference (Detection)](#inference-detection)
- [Switching Between YOLOv10n, YOLO11n, and YOLO11s](#switching-between-yolov10n-yolo11n-and-yolo11s)
- [Results](#results)
- [References](#references)

---

## Introduction

Pothole detection is an essential task for smart city infrastructure and road maintenance. Leveraging the latest YOLO (You Only Look Once) object detection models, this project provides an end-to-end pipeline for training, evaluating, and deploying pothole detectors.

- **YOLOv10n**: Ultralight, fast, and accurate neural network for edge devices.
- **YOLO11n & YOLO11s**: Experimental versions providing improved speed or accuracy. Use the same pipeline—just change the weights file.

---

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- OpenCV (`opencv-python`)
- [YOLOv10 repository code](https://github.com/WongKinYiu/yolov10) or your own implementation
- Other dependencies (see `requirements.txt` if provided)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

1. **Data Format**:  
   Use the YOLO format for annotations:  
   - Each image has a corresponding `.txt` file with object annotations in the format:  
     ```
     <class_id> <x_center> <y_center> <width> <height>
     ```
     (values normalized between 0 and 1)

2. **Directory Structure**:
   ```
   dataset/
     images/
       train/
       val/
     labels/
       train/
       val/
   ```

3. **Classes File**:  
   Create a `data.yaml` (or `classes.names`) file describing your dataset, e.g.:
   ```yaml
   nc: 1
   names: ['pothole']
   ```

---

## Training Process

1. **Download or Prepare Weights**:  
   - For YOLOv10n: [Download YOLOv10n weights](https://github.com/WongKinYiu/yolov10/releases)
   - For YOLO11n/s: Download respective weights or convert from official sources.

2. **Start Training**:
   ```bash
   python train.py \
      --img 640 \
      --batch 16 \
      --epochs 100 \
      --data data.yaml \
      --cfg yolov10n.yaml \
      --weights yolov10n.pt \
      --device 0
   ```
   - Change `--weights` to `yolov11n.pt` or `yolov11s.pt` as needed.

3. **Monitor Training**:  
   - Training logs and results will be saved to the `runs/` directory.

---

## Inference (Detection)

Detect potholes on new images or video:

```bash
python detect.py \
    --weights yolov10n.pt \
    --img 640 \
    --conf 0.25 \
    --source path/to/your/image_or_video
```

- Change `--weights` to the model you want to use (e.g., `yolov11n.pt`).
- Results will be saved in `runs/detect/`.

---

## Switching Between YOLOv10n, YOLO11n, and YOLO11s

You can easily swap the detection backbone in both training and inference:

- **Change Weights File:**
  - For YOLOv10n: `--weights yolov10n.pt`
  - For YOLO11n: `--weights yolov11n.pt`
  - For YOLO11s: `--weights yolov11s.pt`
- **Update Config File if Needed:**  
  Some newer versions may have different `.yaml` config files. Adjust `--cfg` as required.

---

## Results

- Sample detections:
  - Images and videos with bounding boxes around potholes.
- Training metrics:
  - Loss curves, mAP, precision/recall in the `runs/` folder.

---

## References

- [YOLOv10 Official Repository](https://github.com/WongKinYiu/yolov10)
- [YOLOv11n/s (if available)](https://github.com/WongKinYiu/yolov11) *(replace with actual link if public)*
- [PyTorch Documentation](https://pytorch.org/)
- [LabelImg Annotation Tool](https://github.com/tzutalin/labelImg)

---

## Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Note:**  
YOLOv11n and YOLOv11s are experimental and may not be officially released yet. The process described here assumes model weights and config files are compatible with the YOLOv10 training/inference scripts.

---

