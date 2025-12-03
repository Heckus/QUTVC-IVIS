import os
import cv2
import time
import threading
import queue
import argparse
import numpy as np
from collections import deque
from ivis_v1 import IVISDetector  # Importing the package from output 1

# =============================================================================
# --- Configuration ---
# =============================================================================
PROCESSING_WIDTH = 1280
PROCESSING_HEIGHT = 720
BALL_TRAIL_LENGTH = 16

# =============================================================================
# --- Main Processing Thread ---
# =============================================================================
def process_frames(frame_queue, result_queue, ivis_detector, mode, enable_trail):
    """
    Worker thread that uses IVIS_v1 for detection.
    """
    trail_points = deque(maxlen=BALL_TRAIL_LENGTH)

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # --- STEP 1: DETECTION (IVIS Package) ---
        # The detect() function handles YOLO -> Color -> Physics chain internally
        result = ivis_detector.predict(frame)

        # --- STEP 2: VISUALIZATION (IVIS Package) ---
        # Draw the main detection (Box/Circle/Label)
        ivis_detector.visualize(frame, result, mode=mode)

        # --- STEP 3: AESTHETIC TRAIL (Harness Side) ---
        # We keep the trail logic here as it's a visual preference, not core detection
        if enable_trail:
            if result:
                trail_points.append(result['center'])
            
            # Draw trail
            for i in range(1, len(trail_points)):
                if trail_points[i - 1] is None or trail_points[i] is None: continue
                thickness = int(np.sqrt(BALL_TRAIL_LENGTH / float(i + 1)) * 2.5)
                cv2.line(frame, trail_points[i - 1], trail_points[i], (0, 165, 255), thickness)

        result_queue.put(frame)

# =============================================================================
# --- Main Pipeline ---
# =============================================================================
def main(args):
    input_video_path = args.input
    
    if args.output:
        output_video_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_video_path = f"{base_name}_IVIS_mode{args.mode}.mp4"

    # --- Initialize IVIS ---
    print(f"[INFO] Initializing IVIS v1 System...")
    ivis = IVISDetector(model_path=args.model, conf=0.5)

    print(f"[INFO] Opening video: {input_video_path}")
    video = cv2.VideoCapture(input_video_path)
    if not video.isOpened():
        print(f"[ERROR] Could not open video.")
        return

    # Setup Video Writer
    fps_in = video.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_in,
        (PROCESSING_WIDTH, PROCESSING_HEIGHT)
    )
    
    frame_queue = queue.Queue(maxsize=8)
    result_queue = queue.Queue()
    
    print(f"[INFO] Starting processing thread (Mode: {args.mode})...")
    processing_thread = threading.Thread(
        target=process_frames,
        args=(frame_queue, result_queue, ivis, args.mode, not args.no_trail)
    )
    processing_thread.start()

    # --- Frame Reading Loop ---
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = video.read()
        if not ret: break
        
        resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        frame_queue.put(resized_frame)
        
        # Pull processed frames and write to disk
        while not result_queue.empty():
            processed_frame = result_queue.get()
            writer.write(processed_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...", end='\r')

    # Cleanup
    frame_queue.put(None)
    processing_thread.join()

    # Flush remaining frames
    while not result_queue.empty():
        writer.write(result_queue.get())
        frame_count += 1
    
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Saved to: {output_video_path}")
    print(f"Stats: {frame_count} frames in {total_time:.2f}s ({frame_count/total_time:.1f} FPS)")

    video.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run IVIS v1 Detection on Video")
    parser.add_argument("-m", "--model", required=True, help="Path to YOLO .pt file")
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2, 3, 4], 
                        help="1:Box+Label, 2:Box, 3:Circle, 4:Data Only")
    parser.add_argument("--no-trail", action="store_true", help="Disable the orange tail")
    
    args = parser.parse_args()
    main(args)