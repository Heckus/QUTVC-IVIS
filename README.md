# QUT Volleyball Club: Computer Vision & Analytics

**A robust computer vision suite for detecting, tracking, and analyzing volleyballs in match footage.**

Developed by the **Queensland University of Technology (QUT) Volleyball Club**, this repository houses the **IVIS (Integrated Visual Identification System)**—a hybrid tracking engine that combines deep learning, color thresholding, and physics-based momentum prediction to maintain tracking even when traditional detection fails.

---

## Key Features

* **IVIS Architecture:** A three-stage detection pipeline (YOLOv8 $\to$ Color Gating $\to$ Physics Prediction) for robust tracking during motion blur or occlusion.
* **Smart Dataset Tools:** Utilities to merge datasets, extract specific classes from COCO, and split data for training/validation.
* **Interactive Labeler:** A specialized video labeling tool with "Square-Lock" and "Physics-Assist" to speed up manual annotation.
* **Streamlined Training:** One-click training scripts with pre-configured augmentation levels (Light, Medium, Heavy).

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/QUT-Volleyball/vision-analytics.git](https://github.com/QUT-Volleyball/vision-analytics.git)
    cd vision-analytics
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install ultralytics opencv-python numpy pyyaml requests alive-progress tqdm
    ```

---

## The IVIS Engine (`ivis_v1.py`)

The core of this project is the **Integrated Visual Identification System**. Unlike standard YOLO detection which treats every frame independently, IVIS uses a stateful approach:

1.  **Primary Sensor (YOLO):** Attempts to detect the ball using a trained YOLOv8 model.
2.  **Secondary Sensor (Smart Color):** If YOLO fails (e.g., motion blur), it checks for spherical objects matching specific HSV color thresholds. *Note: This is gated by the last known position to prevent false positives across the court.*
3.  **Observer (Momentum Tracker):** If both sensors fail (e.g., total occlusion), a physics engine predicts the ball's trajectory based on recent velocity vectors. It includes "Ghost Frame" logic to stop predicting if the ball is lost for too long.

---

## Usage Workflow

### 1. Inference (Running the Tracker)
To run the full IVIS system on a video:

```bash
python Code/run_ivis.py --model models/best.pt --input match_video.mp4 --mode 1