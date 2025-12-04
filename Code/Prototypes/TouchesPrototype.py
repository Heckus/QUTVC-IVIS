import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
from ivis_v1 import IVISDetector
import sys
import os

# Try importing EasyOCR
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[Warning] EasyOCR not installed.")

class VolleyballAnalytics:
    def __init__(self, source_video, output_video, ivis_model_path='yolo11n.pt'):
        self.source = source_video
        self.output = output_video
        
        if not os.path.exists(source_video):
            print(f"[Error] File not found: {source_video}")
            sys.exit(1)

        # --- Models ---
        print("[System] Loading YOLOv8n (Players)...")
        self.player_model = YOLO('yolov8n.pt') 
        
        print("[System] Loading IVIS (Ball)...")
        self.ball_detector = IVISDetector(model_path=ivis_model_path, conf=0.25)
        
        if OCR_AVAILABLE:
            print("[System] Loading OCR...")
            self.reader = easyocr.Reader(['en'], gpu=True) 

        # --- Game State ---
        self.real_player_stats = defaultdict(int) 
        self.touch_cooldowns = {}
        self.global_touch_cooldown = 0
        
        # Ball Physics State
        self.ball_history = deque(maxlen=6) # Store last 6 positions for vector math
        self.last_ball_vector = None
        
        # --- Identity Registry ---
        # Format: { "Label": { 'hist': h, 'current_track_id': id, 'last_pos': (x,y), 'last_frame': 0 } }
        self.player_registry = {} 
        
        # --- Configs ---
        self.BASE_TOUCH_THRESH = 140
        self.TOUCH_COOLDOWN_FRAMES = 10
        self.GLOBAL_COOLDOWN = 5
        
        # RE-ID Settings
        self.COLOR_SIMILARITY_THRESH = 0.55 
        self.OCR_INTERVAL = 12 
        self.MAX_REID_DISTANCE = 400 # Increased: Players sprint for the ball
        
        # PHYSICS GATES (The "Anti-False-Positive" Layer)
        self.MIN_HIT_ANGLE = 20.0     # Degrees. Ball must deviate by this much to count as a hit.
        self.MIN_VELOCITY_CHANGE = 0.3 # 30% speed change implies a hit/block.
        self.MIN_BALL_SPEED_FOR_HIT = 3.0 # Ignore stationary ball jitters

    def get_color_histogram(self, image, box):
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return None
        
        h, w = roi.shape[:2]
        # Strict Center Crop (Torso)
        center_roi = roi[int(h*0.25):int(h*0.65), int(w*0.3):int(w*0.7)]
        if center_roi.size == 0: return None

        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def read_jersey_number(self, image, box):
        if not OCR_AVAILABLE: return None
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return None
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Top 60% of box for number
        center_slice = gray[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]
        
        if center_slice.size == 0: return None
        try:
            results = self.reader.readtext(center_slice, allowlist='0123456789', detail=0)
            for res in results:
                if res.isdigit() and len(res) <= 2: return res
        except: pass
        return None

    def calculate_vector_change(self):
        """
        Analyzes ball history to detect 'Kinks' in trajectory.
        Returns: (is_kink, angle_deg, speed_ratio)
        """
        if len(self.ball_history) < 5: return False, 0.0, 1.0
        
        # Vector A (Incoming): Frame t-4 to t-2
        p1 = self.ball_history[-5]
        p2 = self.ball_history[-3]
        vec_in = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        
        # Vector B (Outgoing): Frame t-2 to t
        p3 = self.ball_history[-3]
        p4 = self.ball_history[-1]
        vec_out = np.array([p4[0]-p3[0], p4[1]-p3[1]])
        
        norm_in = np.linalg.norm(vec_in)
        norm_out = np.linalg.norm(vec_out)
        
        if norm_in < 2.0 or norm_out < 2.0: return False, 0.0, 1.0 # Too slow
        
        # Calculate Angle Change
        dot_product = np.dot(vec_in, vec_out)
        cos_angle = dot_product / (norm_in * norm_out)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        # Speed Change ratio
        speed_ratio = norm_out / (norm_in + 1e-6)
        
        # Logic: A hit creates an angle change OR a significant speed change
        is_kink = False
        if angle > self.MIN_HIT_ANGLE: is_kink = True
        if speed_ratio > 1.4 or speed_ratio < 0.6: is_kink = True # Bounce or Catch
        
        return is_kink, angle, speed_ratio

    def select_targets(self, cap):
        """User Selection Phase"""
        ret, frame = cap.read()
        if not ret: return

        results = self.player_model.track(frame, persist=True, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy() if results[0].boxes.id is not None else []

        print("\n" + "="*50)
        print(" USER SETUP PHASE")
        print(" 1. Click on player -> Type ID (Name/Number).")
        print(" 2. Press 'q' when done.")
        print("="*50)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_id, clicked_box = None, None
                for i, box in enumerate(boxes):
                    if box[0] < x < box[2] and box[1] < y < box[3]:
                        clicked_id = track_ids[i]
                        clicked_box = box
                        break
                
                if clicked_id is not None:
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                    cv2.imshow("Select Players", frame)
                    cv2.waitKey(1)
                    real_label = input(f">> Identity for YOLO_ID {clicked_id}: ").strip()
                    if real_label:
                        hist = self.get_color_histogram(frame, clicked_box)
                        cx = (clicked_box[0]+clicked_box[2])/2
                        cy = (clicked_box[1]+clicked_box[3])/2
                        self.player_registry[real_label] = {
                            'hist': hist, 
                            'current_track_id': clicked_id,
                            'last_pos': (cx, cy),
                            'last_frame': 0
                        }
                        print(f"   [Registered] {real_label}")

        cv2.namedWindow("Select Players")
        cv2.setMouseCallback("Select Players", mouse_callback)

        while True:
            display = frame.copy()
            active_ids = {p['current_track_id']: l for l, p in self.player_registry.items()}
            for i, box in enumerate(boxes):
                tid = track_ids[i]
                color = (0,255,0) if tid in active_ids else (0,0,255)
                cv2.rectangle(display, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), color, 2)
            cv2.imshow("Select Players", display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cv2.destroyWindow("Select Players")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def reconcile_identities(self, frame, active_tracks, frame_count):
        """
        Robust Re-ID: Only claims new IDs if they are spatially plausible + color match.
        """
        current_yolo_map = {t['id']: t for t in active_tracks}
        assigned_yolo_ids = set()
        
        # 1. Update existing tracks
        for label, profile in self.player_registry.items():
            tid = profile['current_track_id']
            if tid in current_yolo_map:
                assigned_yolo_ids.add(tid)
                box = current_yolo_map[tid]['box']
                profile['last_pos'] = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                profile['last_frame'] = frame_count
                # REMOVED: Auto-adaptation of color (caused drift)
            else:
                profile['current_track_id'] = None

        # 2. Find lost players
        unassigned_tracks = [t for t in active_tracks if t['id'] not in assigned_yolo_ids]
        
        for track in unassigned_tracks:
            t_cx = (track['box'][0] + track['box'][2])/2
            t_cy = (track['box'][1] + track['box'][3])/2
            
            # OCR Check
            if OCR_AVAILABLE and frame_count % self.OCR_INTERVAL == 0:
                det_num = self.read_jersey_number(frame, track['box'])
                if det_num and det_num in self.player_registry:
                    self.player_registry[det_num]['current_track_id'] = track['id']
                    # print(f"[Re-ID] OCR: {det_num}")
                    continue

            # Color + Spatial Check
            track_hist = self.get_color_histogram(frame, track['box'])
            if track_hist is None: continue

            best_lbl, best_score = None, 0
            
            for label, profile in self.player_registry.items():
                if profile['current_track_id'] is None:
                    # Spatial Gate
                    last_pos = profile['last_pos']
                    dist = np.sqrt((t_cx - last_pos[0])**2 + (t_cy - last_pos[1])**2)
                    
                    # Expand search radius over time
                    frames_lost = frame_count - profile['last_frame']
                    search_rad = self.MAX_REID_DISTANCE + (frames_lost * 5)
                    
                    if dist < search_rad:
                        score = cv2.compareHist(profile['hist'], track_hist, cv2.HISTCMP_CORREL)
                        if score > best_score:
                            best_score = score
                            best_lbl = label
            
            if best_lbl and best_score > self.COLOR_SIMILARITY_THRESH:
                self.player_registry[best_lbl]['current_track_id'] = track['id']
                # print(f"[Re-ID] Color: {best_lbl}")

    def get_dynamic_threshold(self, player_y, frame_height):
        # Larger threshold for players closer to camera (bottom)
        norm_y = player_y / frame_height
        return self.BASE_TOUCH_THRESH * max(0.6, norm_y)

    def process(self):
        cap = cv2.VideoCapture(self.source)
        self.select_targets(cap)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))

        frame_idx = 0
        print(f"[System] Processing... (Press 'q' to stop)")

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            # --- 1. Ball Physics ---
            ball_res = self.ball_detector.predict(frame)
            is_trajectory_kink = False
            
            if ball_res:
                self.ball_history.append(ball_res['center'])
                is_kink, angle, speed_ratio = self.calculate_vector_change()
                if is_kink:
                    is_trajectory_kink = True
                    # Visual Debug for Kinks (Yellow Circle)
                    # cv2.circle(frame, ball_res['center'], 20, (0,255,255), 2)
            
            # --- 2. Player Tracking ---
            results = self.player_model.track(frame, persist=True, classes=[0], verbose=False)
            active_tracks = []
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                tids = results[0].boxes.id.int().cpu().numpy()
                for b, tid in zip(boxes, tids):
                    active_tracks.append({'id': tid, 'box': b})
            
            self.reconcile_identities(frame, active_tracks, frame_idx)

            # --- 3. Touch Logic (Physics Gated) ---
            touch_event = None
            if self.global_touch_cooldown > 0: self.global_touch_cooldown -= 1
            
            # Decrement personal cooldowns
            to_del = []
            for lbl in self.touch_cooldowns:
                self.touch_cooldowns[lbl] -= 1
                if self.touch_cooldowns[lbl] <= 0: to_del.append(lbl)
            for lbl in to_del: del self.touch_cooldowns[lbl]

            # CRITICAL LOGIC:
            # Touch = (Proximity) AND (Physics Kink OR Catch/Hold) AND (Not Cooldown)
            if ball_res and self.global_touch_cooldown == 0:
                ball_cx, ball_cy = ball_res['center']
                
                closest_lbl = None
                min_dist = float('inf')
                
                for label, profile in self.player_registry.items():
                    tid = profile['current_track_id']
                    if tid is None: continue
                    
                    # Find track
                    track = next((t for t in active_tracks if t['id'] == tid), None)
                    if not track: continue
                    
                    box = track['box']
                    p_cx = (box[0] + box[2])/2
                    p_cy = (box[1] + box[3])/2
                    
                    dist = np.sqrt((p_cx - ball_cx)**2 + (p_cy - ball_cy)**2)
                    thresh = self.get_dynamic_threshold(p_cy, h)
                    
                    # Vertical Sanity: Ball can't be deep under feet
                    if ball_cy > box[3] + 20: continue

                    if dist < thresh and dist < min_dist:
                        min_dist = dist
                        closest_lbl = label
                
                # If we found a candidate, apply the "Physics Gate"
                if closest_lbl:
                    # RULE: Only count if there is a trajectory kink (Hit) 
                    # OR if the ball is very close to the center of the player (Hold/Set)
                    
                    # Condition A: Physics Event (Kink)
                    valid_hit = is_trajectory_kink
                    
                    # Condition B: Tight Proximity (for soft sets that don't kink much)
                    # If ball is extremely close to head/hands, count it even without big kink
                    if min_dist < 40: valid_hit = True
                    
                    if valid_hit and closest_lbl not in self.touch_cooldowns:
                        self.real_player_stats[closest_lbl] += 1
                        self.touch_cooldowns[closest_lbl] = self.TOUCH_COOLDOWN_FRAMES
                        self.global_touch_cooldown = self.GLOBAL_COOLDOWN
                        touch_event = closest_lbl

            # --- 4. Render ---
            try: frame = self.ball_detector.visualize(frame, ball_res, mode=3)
            except: pass
            
            # Draw Players
            active_yolo_map = {p['current_track_id']: l for l, p in self.player_registry.items()}
            for box, tid in zip(boxes, tids):
                x1,y1,x2,y2 = map(int, box)
                if tid in active_yolo_map:
                    lbl = active_yolo_map[tid]
                    col = (0,255,0)
                    if touch_event == lbl:
                        col = (0,0,255)
                        cv2.putText(frame, "TOUCH!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), col, 2)
                    cv2.putText(frame, str(lbl), (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                else:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (100,100,100), 1)

            # Leaderboard
            cv2.rectangle(frame, (10,10), (200,180), (0,0,0), -1)
            cv2.putText(frame, "STATS", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            y_off = 60
            for l, s in sorted(self.real_player_stats.items(), key=lambda x:x[1], reverse=True)[:5]:
                cv2.putText(frame, f"{l}: {s}", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                y_off += 25

            out.write(frame)
            cv2.imshow("Process", frame)
            if cv2.waitKey(1) == ord('q'): break
            if frame_idx % 20 == 0: sys.stdout.write(f"\rFrame {frame_idx}"); sys.stdout.flush()

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Save CSV
        pd.DataFrame([{"Player":k, "Touches":v} for k,v in self.real_player_stats.items()]).to_csv(self.output.replace(".mp4",".csv"), index=False)
        print("\nDone.")

if __name__ == "__main__":
    pipeline = VolleyballAnalytics(
        source_video='S:/Datasets/ball5datasets/newdata/raw/3.mp4', 
        output_video='output_final.mp4',
        ivis_model_path='models/modeln_ball2.pt'
    )
    pipeline.process()
    print("Pipeline Ready: Supports Color + OCR + Interactive ID Assignment.")

