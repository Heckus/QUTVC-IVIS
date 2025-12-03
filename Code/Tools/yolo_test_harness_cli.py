# yolo_test_harness_cli.py
# A flexible script to test object detection of a YOLOv8 .pt model on a video.
# Model and video paths are provided as command-line arguments.
# Includes an optional aesthetic trail for ball tracking.

# =============================================================================
# --- Imports ---
# =============================================================================
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
import argparse
from collections import deque # Added for ball trail

# =============================================================================
# --- Configuration ---
# =============================================================================
# --- Detection Settings ---
YOLO_CONF_THRESHOLD = 0.6
CLASSES_TO_DETECT = [0] # Example: [0, 32] for person and sports ball in COCO

# --- Ball Trail Settings ---
# Set to True to enable the visual trail for the ball
ENABLE_BALL_TRAIL = True
# The class name of the ball in your YOLO model. This MUST match the name in your model's .yaml file.
# The default for the standard COCO model is "sports ball".
BALL_CLASS_NAME = "volleyball"
# The number of frames the trail will last.
BALL_TRAIL_LENGTH = 16

# --- Processing Settings ---
PROCESSING_WIDTH = 1280
PROCESSING_HEIGHT = 720

# =============================================================================
# --- Main Processing Thread ---
# =============================================================================
def process_frames(frame_queue, result_queue, yolo_model):
    """
    Worker thread function to perform YOLO detection and draw visuals on frames.
    """
    # Initialize a deque to store the center points of the ball for the trail
    if ENABLE_BALL_TRAIL:
        trail_points = deque(maxlen=BALL_TRAIL_LENGTH)

    while True:
        frame = frame_queue.get()
        if frame is None:  # Sentinel value to signal the end
            break

        # Perform YOLO inference
        results = yolo_model.predict(
            frame,
            conf=YOLO_CONF_THRESHOLD,
            classes=CLASSES_TO_DETECT,
            verbose=False
        )
        
        result = results[0]
        class_names = result.names

        # Iterate through each detected object in the frame
        for box in result.boxes:
            coords = [int(i) for i in box.xyxy[0]]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = class_names.get(cls_id, f"Class {cls_id}")

            # --- Ball Trail Logic ---
            # If ball tracking is enabled and the detected object is a ball, add its center to the trail
            if ENABLE_BALL_TRAIL and class_name == BALL_CLASS_NAME:
                x1_b, y1_b, x2_b, y2_b = coords
                center = ((x1_b + x2_b) // 2, (y1_b + y2_b) // 2)
                trail_points.append(center)

            # --- Drawing Logic ---
            x1, y1, x2, y2 = coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{class_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # --- Draw the Ball Trail ---
        if ENABLE_BALL_TRAIL:
            for i in range(1, len(trail_points)):
                if trail_points[i - 1] is None or trail_points[i] is None:
                    continue
                # Calculate thickness to make the trail fade out
                thickness = int(np.sqrt(BALL_TRAIL_LENGTH / float(i + 1)) * 2.5)
                # Draw a line segment for the trail
                cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 165, 255), thickness)

        result_queue.put(frame)

# =============================================================================
# --- Main Pipeline ---
# =============================================================================
def main(args):
    """
    Main function to set up the video processing pipeline.
    """
    input_video_path = args.input
    yolo_model_path = args.model
    
    if args.output:
        output_video_path = args.output
    else:
        base_name, ext = os.path.splitext(os.path.basename(input_video_path))
        output_dir = os.path.dirname(input_video_path) if os.path.dirname(input_video_path) else '.'
        output_video_path = os.path.join(output_dir, f"{base_name}_output.mp4")

    print(f"[INFO] Loading YOLO model: {yolo_model_path}")
    if not os.path.exists(yolo_model_path):
        print(f"[ERROR] YOLO model file not found at: {yolo_model_path}")
        return
    yolo_model = YOLO(yolo_model_path)

    print(f"[INFO] Opening video file: {input_video_path}")
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"[ERROR] Could not open video file: {input_video_path}")
        return

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        video.get(cv2.CAP_PROP_FPS),
        (PROCESSING_WIDTH, PROCESSING_HEIGHT)
    )
    
    frame_queue = queue.Queue(maxsize=8)
    result_queue = queue.Queue()
    
    print("[INFO] Starting processing thread...")
    processing_thread = threading.Thread(
        target=process_frames,
        args=(frame_queue, result_queue, yolo_model)
    )
    processing_thread.start()

    frame_count = 0
    start_time = time.time()
    
    print("[INFO] Reading frames from video...")
    while True:
        ret, frame = video.read()
        if not ret: break
        #frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotate frame if needed ========================================================================
        resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        frame_queue.put(resized_frame)
        
        if not result_queue.empty():
            processed_frame = result_queue.get()
            writer.write(processed_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"[INFO] Processed {frame_count} frames...")

    print("[INFO] End of video reached. Cleaning up...")
    frame_queue.put(None)
    processing_thread.join()

    while not result_queue.empty():
        writer.write(result_queue.get())
        frame_count += 1
    
    end_time = time.time()
    processing_time = end_time - start_time
    fps = frame_count / processing_time if processing_time > 0 else 0

    print(f"\n[INFO] Processing complete.")
    print(f"Output video saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}, Total time: {processing_time:.2f}s, Estimated FPS: {fps:.2f}")

    video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video with a YOLOv8 model to detect objects and optionally draw a trail for a specific class.")
    
    parser.add_argument("-m", "--model", required=True, help="Path to the YOLOv8 .pt model file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", help="Optional. Path to save the output video file. If not provided, it will be saved next to the input file with an '_output' suffix.")
    
    args = parser.parse_args()
    
    main(args)