import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import yaml
import math

# =============================================================================
# --- CLASS: Momentum Tracker (Physics Prediction) ---
# =============================================================================
class MomentumTracker:
    def __init__(self):
        self.history = []  # Stores center points [(x,y), (x,y), ...]
        self.max_history = 5 # Number of frames to calculate velocity from
    
    def update(self, box):
        """Add a new position to the history."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        self.history.append((cx, cy))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def replace_last(self, box):
        """
        Control Theory 'Correction' Step:
        Overwrites the last entry. Used when the User (Ground Truth) 
        corrects the YOLO (Sensor) detection.
        """
        if self.history:
            self.history.pop()
        self.update(box)
            
    def predict(self, frame_shape):
        """Predict next position based on velocity vector."""
        if len(self.history) < 2:
            return None
        
        # Calculate velocity vector (dx, dy)
        (x1, y1) = self.history[-2]
        (x2, y2) = self.history[-1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Predicted center
        pred_x = int(x2 + dx)
        pred_y = int(y2 + dy)
        
        # Bounds check
        h, w = frame_shape[:2]
        if 0 <= pred_x < w and 0 <= pred_y < h:
            return (pred_x, pred_y)
        return None
    
    def reset(self):
        self.history = []

#=============================================================================
# --- CLASS: Smart Color Detector (Roundness + Square Lock + Padding) ---
# =============================================================================
class ColorBlobDetector:
    def __init__(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.lower_yellow = np.array([18, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.frame_count = 0
        self.warmup_frames = 50
        
        # --- NEW PARAMETER ---
        # 0.0 = exact fit to color contour
        # 0.2 = 20% larger than the contour (Recommended)
        self.box_padding = 0.2 

    def detect(self, frame):
        self.frame_count += 1
        fgMask = self.backSub.apply(frame)
        if self.frame_count < self.warmup_frames:
            return []

        h_img, w_img = frame.shape[:2]
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        colorMask = cv2.inRange(hsv_frame, self.lower_yellow, self.upper_yellow)
        combinedMask = cv2.bitwise_and(fgMask, colorMask)
        kernel = np.ones((11, 11), np.uint8)
        closed_mask = cv2.morphologyEx(combinedMask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        detected_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100 or area > 20000:
                continue
            
            # --- Roundness Check ---
            ((cx, cy), radius) = cv2.minEnclosingCircle(c)
            circle_area = np.pi * (radius ** 2)
            if circle_area == 0: continue
            
            roundness = area / circle_area
            if roundness < 0.60:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # --- Force Square & Apply Padding ---
            center_x = x + w // 2
            center_y = y + h // 2
            
            # 1. Determine base size (largest dimension)
            side_length = max(w, h)
            
            # 2. Apply expansion factor
            # We multiply by (1 + padding) to grow the box relative to its size
            expanded_side = side_length * (1 + self.box_padding)
            r = int(expanded_side / 2)
            
            # 3. Clamp to screen boundaries
            new_x1 = max(0, center_x - r)
            new_y1 = max(0, center_y - r)
            new_x2 = min(w_img, center_x + r)
            new_y2 = min(h_img, center_y + r)

            detected_boxes.append((new_x1, new_y1, new_x2, new_y2))
            
        return detected_boxes
# =============================================================================
# --- Global Variables & Mouse Callback ---
# =============================================================================
drawing = False
ix, iy = -1, -1
WINDOW_NAME = 'Volleyball Labeler'

def multi_box_editor_callback(event, x, y, flags, param):
    global ix, iy, drawing
    boxes = param['boxes']
    frame_copy = param['frame'].copy()
    h_img, w_img, _ = frame_copy.shape
    source_map = param['source_map']
    
    # Visual Params
    physics_enabled = param.get('physics_enabled', True)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw_interface(frame_copy, boxes, source_map, physics_enabled)
            
            # Square Lock Logic
            r = max(abs(x - ix), abs(y - iy))
            x1 = max(0, ix - r)
            y1 = max(0, iy - r)
            x2 = min(w_img, ix + r)
            y2 = min(h_img, iy + r)
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame_copy, (ix, iy), 3, (0, 0, 255), -1)
            cv2.imshow(WINDOW_NAME, frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        r = max(abs(x - ix), abs(y - iy))
        x1 = max(0, ix - r)
        y1 = max(0, iy - r)
        x2 = min(w_img, ix + r)
        y2 = min(h_img, iy + r)

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
            draw_interface(frame_copy, boxes, source_map, physics_enabled)
            cv2.imshow(WINDOW_NAME, frame_copy)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if boxes:
            min_dist, box_to_delete_idx = float('inf'), -1
            for i, box in enumerate(boxes):
                center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                dist = math.sqrt((center_x - x)**2 + (center_y - y)**2)
                if dist < min_dist:
                    min_dist, box_to_delete_idx = dist, i
            
            if box_to_delete_idx != -1:
                box_width = boxes[box_to_delete_idx][2] - boxes[box_to_delete_idx][0]
                if min_dist < max(box_width, 50): 
                    boxes.pop(box_to_delete_idx)
                    draw_interface(frame_copy, boxes, source_map, physics_enabled)
                    cv2.imshow(WINDOW_NAME, frame_copy)

# =============================================================================
# --- Helper Functions ---
# =============================================================================
def draw_interface(frame, boxes, source_map, physics_enabled):
    # Draw Boxes
    for i, box in enumerate(boxes):
        color = source_map.get(i, (0, 255, 0)) 
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    
    # Draw Physics Indicator (Top Left)
    status_text = "PHYSICS: ON" if physics_enabled else "PHYSICS: OFF (RESET)"
    status_color = (0, 255, 0) if physics_enabled else (0, 0, 255)
    
    # Background for text to make it readable
    cv2.rectangle(frame, (5, 5), (280, 35), (50, 50, 50), -1)
    cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, status_color, 2)

def convert_to_yolo_format(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    dw, dh = 1.0 / img_width, 1.0 / img_height
    x_center, y_center = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width, height = x2 - x1, y2 - y1
    return f"0 {x_center*dw:.6f} {y_center*dh:.6f} {width*dw:.6f} {height*dh:.6f}"

def save_labels_and_image(frame, bboxes, base_filename, dirs, target_dims):
    target_width, target_height = target_dims
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    image_path = os.path.join(dirs['images'], f"{base_filename}.jpg")
    cv2.imwrite(image_path, resized_frame)
    
    label_path = os.path.join(dirs['labels'], f"{base_filename}.txt")
    if not bboxes:
        open(label_path, 'w').close()
        return True

    orig_height, orig_width, _ = frame.shape
    yolo_labels = []
    for bbox in bboxes:
        yolo_labels.append(convert_to_yolo_format(bbox, orig_width, orig_height))

    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_labels))
    return True

def print_instructions():
    print("\n--- Volleyball Interactive Labeler (Physics & Square Lock) ---")
    print("  's' - Save all current boxes and update Physics Tracker")
    print("  'd' - Discard frame")
    print("  'c' - Clear all boxes")
    print("  'q' - Quit")
    print("\n--- Controls ---")
    print("  Left-Click & Drag - Create SQUARE box from CENTER")
    print("  'p'      - Toggle Physics Prediction ON/OFF")
    print("\n--- Box Colors ---")
    print("  BLUE   - YOLO Detection")
    print("  YELLOW - Color Detection (Roundness Filtered)")
    print("  PURPLE - Physics Prediction (Momentum)")
    print("  GREEN  - Manual")
    print("----------------------------------------------------------\n")

# =============================================================================
# --- Main Function ---
# =============================================================================
def main(args):
    TARGET_DIMS = (640, 640)
    YOLO_MODEL = "../../models/modeln_ball2.pt" 
    
    print_instructions()
    
    try:
        yolo_model = YOLO(YOLO_MODEL)
    except Exception as e:
        print(f"[WARNING] YOLO model load failed: {e}. Proceeding without YOLO.")
        yolo_model = None

    assistant_detector = ColorBlobDetector()
    tracker = MomentumTracker()
    
    video_path, output_dir = args.video, args.output_dir
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    dirs = {'images': os.path.join(output_dir, "images"), 'labels': os.path.join(output_dir, "labels")}
    os.makedirs(dirs['images'], exist_ok=True); os.makedirs(dirs['labels'], exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"[ERROR] Could not open video file: {video_path}"); return
    
    frame_idx, saved_frame_count = 0, 0
    last_w, last_h = 50, 50 
    
    # Toggles
    physics_enabled = True 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        current_boxes = []
        source_map = {}
        auto_tracked_this_frame = False # Flag to track if we did a sensor update
        
        # --- 1. SENSOR DETECTIONS (YOLO / Color) ---
        if yolo_model:
            yolo_results = yolo_model.predict(frame, conf=args.conf, classes=[0], verbose=False)
            current_boxes = [ [int(i) for i in box.xyxy[0]] for box in yolo_results[0].boxes ]
            if current_boxes:
                source_map = {i: (255, 100, 0) for i in range(len(current_boxes))} # Blue

        if not current_boxes:
            assistant_boxes = assistant_detector.detect(frame)
            if assistant_boxes:
                current_boxes = assistant_boxes
                source_map = {i: (0, 255, 255) for i in range(len(current_boxes))} # Yellow

        # --- 2. SENSOR UPDATE (Immediate Physics Influence) ---
        if current_boxes and physics_enabled:
            # We use the first detection to guide the physics engine IMMEDIATELY
            tracker.update(current_boxes[0])
            auto_tracked_this_frame = True
            # Update last known size
            last_w = current_boxes[0][2] - current_boxes[0][0]
            last_h = current_boxes[0][3] - current_boxes[0][1]

        # --- 3. PHYSICS PREDICTION (If Sensors Failed) ---
        if not current_boxes and physics_enabled:
            predicted_center = tracker.predict(frame.shape)
            if predicted_center:
                px, py = predicted_center
                r = max(last_w, last_h) // 2
                h_img, w_img = frame.shape[:2]
                
                p_x1 = max(0, px - r)
                p_y1 = max(0, py - r)
                p_x2 = min(w_img, px + r)
                p_y2 = min(h_img, py + r)
                
                current_boxes.append((p_x1, p_y1, p_x2, p_y2))
                source_map = {0: (255, 0, 255)} # Purple

        cv2.namedWindow(WINDOW_NAME)
        # Pass physics_enabled to callback so it can draw the indicator
        callback_param = {
            'boxes': current_boxes, 
            'frame': frame, 
            'source_map': source_map,
            'physics_enabled': physics_enabled
        }
        cv2.setMouseCallback(WINDOW_NAME, multi_box_editor_callback, callback_param)

        while True:
            frame_copy = frame.copy()
            draw_interface(frame_copy, current_boxes, source_map, physics_enabled)
            
            info_text = f"Frame: {frame_idx} | Saved: {saved_frame_count} | 'p' Toggle Physics"
            cv2.putText(frame_copy, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow(WINDOW_NAME, frame_copy)

            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return
                
            elif key == ord('p'):
                physics_enabled = not physics_enabled
                # Update param so mouse callback knows new state immediately
                callback_param['physics_enabled'] = physics_enabled
                if not physics_enabled:
                    tracker.reset() # RESET HISTORY
                    print("Physics Tracker RESET.")
                else:
                    print("Physics Tracker ENABLED.")
                    
            elif key == ord('s'):
                base_filename = f"{video_filename}_frame_{frame_idx:06d}"
                if save_labels_and_image(frame, current_boxes, base_filename, dirs, TARGET_DIMS):
                    saved_frame_count += 1
                    print(f"Saved frame {frame_idx}")
                    
                    # --- 4. GROUND TRUTH CORRECTION ---
                    # If we have a finalized box (manually confirmed or edited)
                    if current_boxes and physics_enabled:
                        if auto_tracked_this_frame:
                            # If we ALREADY updated based on YOLO/Color, but the user might have
                            # moved the box, we perform a CORRECTION (replace last entry).
                            tracker.replace_last(current_boxes[0])
                        else:
                            # If it was a purely manual add (physics/sensors didn't find it),
                            # we treat this as a new update.
                            tracker.update(current_boxes[0])
                            
                        # Keep size updated
                        last_w = current_boxes[0][2] - current_boxes[0][0]
                        last_h = current_boxes[0][3] - current_boxes[0][1]
                break
                
            elif key == ord('d'):
                print(f"Discarded frame {frame_idx}")
                break
            
            elif key == ord('c'):
                current_boxes.clear()
                source_map.clear()
        
        frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
    # Save YAML
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_output_dir = os.path.abspath(output_dir)
    yaml_data = {'path': abs_output_dir, 'train': 'images', 'val': 'images', 'names': {0: 'volleyball'}}
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"[SUCCESS] Dataset ready in: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Volleyball Labeler (Square Lock + Physics)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence threshold.")
    args = parser.parse_args()
    main(args)