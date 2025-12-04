import cv2
import numpy as np
from ultralytics import YOLO
from engine import BallKalmanFilter, NetHomography, PersistentPlayerTracker, calculate_iou
from ivis_v1 import IVISDetector 
from collections import defaultdict, deque
import sys
import os

# --- CONFIGURATION ---

SOURCE_VIDEO = 'S:/Datasets/ball5datasets/newdata/raw/3.mp4'
BALL_MODEL_PATH = 'models/modeln_ball2.pt'
# MODEL 1: STRONG TRACKER (Location/ID)
TRACKING_MODEL_PATH = 'yolov8s.pt' # Using Extra Large for max robustness
# MODEL 2: POSE (Touches)
POSE_MODEL_PATH = 'yolov8n-pose.pt'
OUTPUT_NAME = 'processed_volleyball_ivis_pro.mp4'
DEBUG_MODE = True
class VolleyballSystem:
    def __init__(self):
        print("[Init] Loading Models...")
        print(f"   1. Tracker: {TRACKING_MODEL_PATH}")
        self.tracking_model = YOLO(TRACKING_MODEL_PATH)
        
        print(f"   2. Pose: {POSE_MODEL_PATH}")
        self.pose_model = YOLO(POSE_MODEL_PATH)
        
        print(f"   3. Ball (IVIS): {BALL_MODEL_PATH}")
        self.ivis = IVISDetector(model_path=BALL_MODEL_PATH, conf=0.25)
        
        # Physics & Geometry
        self.ball_kf = BallKalmanFilter() 
        self.net_geo = NetHomography()
        self.player_tracker = PersistentPlayerTracker()
        
        # Game State
        self.touch_stats = defaultdict(int)
        self.ball_trajectory = deque(maxlen=10) 
        self.velocity_history = deque(maxlen=10)
        
        self.frame_count = 0
        self.cooldown = 0
        self.calibration_done = False
        self.players_calibrated = False

    def calibrate_net(self, cap):
        """ Human-in-the-Loop: Net Definition """
        print("\n" + "="*50)
        print(" PHASE 1: NET CALIBRATION")
        print(" 1. Top-Left (Net Tape)")
        print(" 2. Top-Right (Net Tape)")
        print(" 3. Bottom-Right (Ground)")
        print(" 4. Bottom-Left (Ground)")
        print("="*50)
        
        ret, frame = cap.read()
        if not ret: return
        points = []
        
        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_cb)
        
        while True:
            display = frame.copy()
            cv2.putText(display, f"Net Points: {len(points)}/4", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            for i, p in enumerate(points):
                cv2.circle(display, p, 5, (0, 0, 255), -1)
                if i > 0: cv2.line(display, points[i-1], p, (0, 255, 0), 2)
            
            if len(points) == 4:
                cv2.line(display, points[3], points[0], (0, 255, 0), 2)
                cv2.line(display, points[3], points[2], (255, 0, 0), 3)
                cv2.putText(display, "Press 'q' to Confirm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Calibration", display)
            if cv2.waitKey(1) & 0xFF == ord('q') and len(points) == 4:
                self.net_geo.compute_matrix(points)
                self.calibration_done = True
                break
        
        # No destroy here, keep window for next phase

    def calibrate_players(self, cap):
        """ Select 6 Players using the STRONG TRACKER """
        print("\n" + "="*50)
        print(" PHASE 2: PLAYER SELECTION")
        print(" Click on the 6 players to track.")
        print("="*50)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        
        # USE STRONG MODEL FOR CALIBRATION
        results = self.tracking_model.track(frame, persist=True, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        selected_count = 0
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal selected_count
            if event == cv2.EVENT_LBUTTONDOWN:
                for box in boxes:
                    if box[0] < x < box[2] and box[1] < y < box[3]:
                        selected_count += 1
                        label = f"P{selected_count}"
                        self.player_tracker.register_player(label, box)
                        print(f"   Registered {label}")
                        break

        cv2.setMouseCallback("Calibration", mouse_cb)

        while True:
            display = frame.copy()
            for box in boxes:
                cv2.rectangle(display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (100, 100, 100), 1)

            for label, p_data in self.player_tracker.tracks.items():
                b = p_data['box']
                c = p_data['color']
                cv2.rectangle(display, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
                cv2.putText(display, label, (int(b[0]), int(b[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

            cv2.putText(display, f"Selected: {selected_count}/6", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if selected_count >= 6:
                cv2.putText(display, "Press 'q' to START", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Calibration", display)
            if cv2.waitKey(1) & 0xFF == ord('q') and selected_count > 0:
                self.players_calibrated = True
                break
        
        cv2.destroyWindow("Calibration")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def detect_hit_event(self):
        """ Detects hit based on velocity change (Calculated by KF) """
        if len(self.velocity_history) < 4: return False, 0.0
        
        v_curr = np.array(self.velocity_history[-1])
        v_prev = np.array(self.velocity_history[-4])
        mag_curr = np.linalg.norm(v_curr)
        mag_prev = np.linalg.norm(v_prev)
        
        if mag_curr < 2.0 or mag_prev < 2.0: return False, 0.0
        cos_sim = np.dot(v_curr, v_prev) / (mag_curr * mag_prev)
        
        if cos_sim < 0.8: return True, mag_curr
        return False, 0.0

    def process_video(self):
        cap = cv2.VideoCapture(SOURCE_VIDEO)
        if not cap.isOpened(): return

        # --- SETUP PHASES ---
        self.calibrate_net(cap)
        if not self.calibration_done: return
        self.calibrate_players(cap)
        if not self.players_calibrated: return

        w = int(cap.get(3))
        h = int(cap.get(4))
        out = cv2.VideoWriter(OUTPUT_NAME, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        print("[System] Tracking 6 Players (Dual Fusion Engine)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            self.frame_count += 1
            display = frame.copy()

            # --- 1. DUAL MODEL INFERENCE ---
            
            # A. Strong Tracker (Drives the Boxes)
            track_results = self.tracking_model.track(frame, persist=True, classes=[0], verbose=False, conf=0.1)
            t_boxes = track_results[0].boxes.xyxy.cpu().numpy() if track_results[0].boxes.id is not None else []
            
            # B. Pose Model (Drives the Wrists)
            pose_results = self.pose_model.predict(frame, verbose=False, conf=0.1)
            p_boxes = pose_results[0].boxes.xyxy.cpu().numpy() if pose_results[0].boxes else []
            p_kps = pose_results[0].keypoints.data.cpu().numpy() if pose_results[0].keypoints is not None else []
            
            # --- 2. SENSOR FUSION ---
            fused_detections = []
            
            for t_box in t_boxes:
                # Filter: Must be on close side
                feet_y = int(t_box[3])
                feet_x = int((t_box[0]+t_box[2])/2)
                if not self.net_geo.is_player_on_close_side(feet_x, feet_y):
                    continue
                
                # Find matching Pose Box (IoU Match)
                best_kps = None
                best_iou = 0
                
                for idx, p_box in enumerate(p_boxes):
                    iou = calculate_iou(t_box, p_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_kps = p_kps[idx]
                
                # Fusion: Use Strong Box, Attach Pose Keypoints if IoU > 0.4
                final_kps = best_kps if best_iou > 0.4 else None
                fused_detections.append({'box': t_box, 'kps': final_kps})

            # --- 3. PERSISTENT TRACKING ---
            active_players = self.player_tracker.update(fused_detections)

            # Draw Players
            for label, p_data in active_players.items():
                box = p_data['box']
                color = p_data['color']
                kps = p_data['kps']
                
                # Draw Box
                cv2.rectangle(display, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(display, label, (int(box[0]), int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw Wrists
                if kps is not None:
                    for k_idx in [9, 10]: 
                        if kps[k_idx][2] > 0.3:
                            wx, wy = int(kps[k_idx][0]), int(kps[k_idx][1])
                            cv2.circle(display, (wx, wy), 4, (0, 255, 255), -1)

            # --- 4. BALL & TOUCH ---
            ivis_result = self.ivis.predict(frame)
            ball_detected = False
            bx, by = 0, 0
            
            if ivis_result:
                bx, by = ivis_result['center']
                ball_detected = True
                self.ball_kf.update(bx, by)
                vx, vy = self.ball_kf.get_velocity()
                
                self.ball_trajectory.append((bx, by))
                self.velocity_history.append((vx, vy))
                
                frame = self.ivis.visualize(display, ivis_result, mode=3)
                
                real_h = self.net_geo.get_real_height(bx, by)
                h_text = f"{real_h:.2f}m"
                cv2.putText(display, h_text, (bx+12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
                cv2.putText(display, h_text, (bx+12, by), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Touch Logic
            if self.cooldown > 0: self.cooldown -= 1
            if ball_detected and self.cooldown == 0:
                is_physics_hit, intensity = self.detect_hit_event()
                hit_candidate = None
                min_dist = 1000
                
                for label, p_data in active_players.items():
                    if p_data['missing_frames'] > 5: continue
                    kps = p_data['kps']
                    p_cx, p_cy = p_data['center']
                    
                    # Wrist
                    wrist_close = False
                    if kps is not None:
                        for k_idx in [9, 10]:
                            if kps[k_idx][2] > 0.3:
                                d = np.hypot(bx - kps[k_idx][0], by - kps[k_idx][1])
                                if d < 45: wrist_close = True
                    
                    # Decision
                    is_touch = False
                    if wrist_close:
                        if is_physics_hit or intensity > 2.0: is_touch = True
                    
                    body_dist = np.hypot(bx - p_cx, by - p_cy)
                    if is_touch and body_dist < min_dist:
                        min_dist = body_dist
                        hit_candidate = label

                if hit_candidate is not None:
                    self.touch_stats[hit_candidate] += 1
                    self.cooldown = 15
                    cv2.putText(display, f"TOUCH: {hit_candidate}", (bx, by-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # --- STATS ---
            y_offset = 50
            for pid, count in sorted(self.touch_stats.items(), key=lambda x:x[1], reverse=True):
                col = self.player_tracker.tracks[pid]['color']
                cv2.putText(display, f"{pid}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                y_offset += 25

            out.write(display)
            if DEBUG_MODE:
                cv2.imshow('Volleyball Dual Fusion', display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Done. Saved to {OUTPUT_NAME}")

if __name__ == "__main__":
    system = VolleyballSystem()
    system.process_video()