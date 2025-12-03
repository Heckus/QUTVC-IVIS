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

--mode 1: Draws Bounding Box + Label + Confidence Source.

--mode 3: Draws a minimal Circle (great for overlay graphics).

--no-trail: Disables the aesthetic orange motion trail.
```

2. Dataset Creation & Labeling

Create your own datasets from raw match footage using the interactive labeler. This tool includes a physics-assist mode that predicts where the box should be, speeding up your workflow.
```bash
python Code/Tools/create_dataset_from_video.py --video raw_match.mp4 --output-dir ./new_dataset
```
    Controls:

        Left Click + Drag: Draw a square-locked box from the center out.

        p: Toggle Physics Prediction (Let the code guess the next frame for you).

        s: Save frame and advance.

        d: Discard frame.

3. Dataset Management

    Extract from COCO: Grab thousands of "sports ball" images to bootstrap your model.
    
```bash
python Code/Tools/COCOdatasetextractor.py annotations.json -t 1000 -v 200 -cat "sports ball"
```
Merge Datasets: Combine multiple datasets into one master set.
```bash
python Code/Tools/smartdatasetmerger.py --datasets data1.yaml data2.yaml --output ./merged_data
```
Split Data: Automatically split images into Train/Val folders.
```bash

    python Code/Tools/split_dataset.py --data-dir ./my_data --train-ratio 0.8
```
4. Training

Train a YOLOv8 model with custom augmentation presets.
```bash

# Options: light, medium, heavy, none
python Code/Tools/trainyolo.py --data data.yaml --model yolov8n.pt --epochs 100 --aug medium
```
File Structure

    Code/ivis_v1.py: The core detection package containing the MomentumTracker and ColorBlobDetector classes.

    Code/run_ivis.py: The video inference harness that utilizes IVIS.

    Code/Tools/:

        create_dataset_from_video.py: Interactive labeling tool.

        trainyolo.py: Training wrapper with augmentation presets.

        split_dataset.py: Utility to split raw images into Train/Val sets.

        smartdatasetmerger.py: Utility to combine datasets.

        COCOdatasetextractor.py: Utility to download specific classes from COCO.

        yolo_test_harness_cli.py: A basic YOLO-only viewer (for comparing baseline vs IVIS).

License

This project is maintained by the Queensland University of Technology Volleyball Club.

Happy Spiking! 🏐