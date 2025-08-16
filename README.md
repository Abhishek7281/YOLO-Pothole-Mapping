# Pothole Detection using YOLOv10n, YOLO11n, and YOLO11s

This repository contains the implementation of our study on **pothole detection and mapping** using recent YOLO models (**YOLOv10n, YOLO11n, YOLO11s**).  
The work was conducted to provide a **lightweight, real-time, and cost-effective framework** for infrastructure monitoring.

---

## 📌 Features
- Implementation of **YOLOv10n, YOLO11n, and YOLO11s** for pothole detection.
- Pre-trained weight files included in the repository under `project_files/`.
- Easy-to-use **Streamlit app (`app.py`)** for interactive detection and visualization.
- Supports switching between YOLOv10n, YOLO11n, and YOLO11s by replacing the weight file path.
- Includes results of all three models for reproducibility.
- Integrated with **GPS and GIS mapping** for accurate localization (described in manuscript).

---

## 📂 Repository Structure
```

├── project\_files/        # Contains pre-trained YOLO weight files
├── utils/                # Utility functions
├── .streamlit/           # Streamlit configuration
├── app.py                # Streamlit app for pothole detection
├── requirements.txt      # Python dependencies
├── packages.txt
└── README.md             # Documentation

````

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/Abhishek7281/Potholes-Detection-Using-YOLOV10n-YOLO11n-and-YOLO11s-.git
cd Potholes-Detection-Using-YOLOV10n-YOLO11n-and-YOLO11s-
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Streamlit app:

```bash
streamlit run app.py
```

* By default, the app runs with **YOLOv10n** weights.
* To use **YOLO11n** or **YOLO11s**, replace the weight file in `project_files/` and update the path in `app.py`.

Example (inside `app.py`):

```python
# Change this line to point to your desired YOLO weights
weights_path = "project_files/yolov11n.pt"
```

---

## 📊 Results

* Results for **YOLOv10n, YOLO11n, and YOLO11s** are available in this repository.
* YOLO11n achieved the **best trade-off** between accuracy, inference time (1.1 ms), and model size (5.6 MB).

---

## 📁 Dataset

* The dataset used in this study is publicly available at the following link:
Google Drive Dataset Link : https://drive.google.com/file/d/1qEBV9wqBzuLbUBFnV_GBROCJLSoygrUT/view

This dataset was prepared using the RoboFlow platform and contains 13,767 images with corresponding annotations for pothole detection. It can be directly used for training and evaluating YOLOv10n, YOLO11n, and YOLO11s models.

---

## 📜 Citation

If you use this repository, please cite our paper:

```
@article{your_paper_citation,
  title   = {The Application of YOLO Models in Detecting Potholes Ensures Fast, Accurate, and Cost-Effective Solutions for Infrastructure Monitoring},
  author  = {Abhishek Kumar Pathak et al.},
  year    = {2025},
  journal = {Journal Name}
}
```

---

## 🙏 Acknowledgement

We thank the reviewers for their valuable suggestions to make this repository open and reproducible for the research community.
