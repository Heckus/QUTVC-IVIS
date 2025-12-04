import cv2
import numpy as np
from ultralytics import YOLO
import math

# =============================================================================
# --- COMPONENT 1: MOMENTUM TRACKER (Physics) ---
# =============================================================================
# =============================================================================
# --- COMPONENT 1: MOMENTUM TRACKER (Physics) ---
# =============================================================================
class MomentumTracker:
    def __init__(self, max_history=5, max_ghost_frames=2, max_speed=100):
        """
        max_speed: Maximum pixels the ball is allowed to travel in a single frame.
                   This acts as the radius constraint.
        """
        self.history = [] 
        self.max_history = max_history
        self.max_ghost_frames = max_ghost_frames
        self.max_speed = max_speed # <--- NEW PARAMETER
        self.ghost_counter = 0 
        self.last_dims = (50, 50) 

    def update(self, box):
        """Called when a REAL sensor (YOLO/Color) finds the ball."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        self.history.append((cx, cy))
        
        w = box[2] - box[0]
        h = box[3] - box[1]
        self.last_dims = (w, h)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self.ghost_counter = 0

    def get_last_position(self):
        """Returns the last known (x,y) or None."""
        if not self.history: return None
        return self.history[-1]

    def predict(self, frame_shape):
        if self.ghost_counter >= self.max_ghost_frames:
            return None

        if len(self.history) < 2:
            return None
        
        # 1. Calculate raw velocity vector
        (x1, y1) = self.history[-2]
        (x2, y2) = self.history[-1] # Last known real position
        dx = x2 - x1
        dy = y2 - y1
        
        # 2. Calculate magnitude of the movement (speed)
        speed = math.sqrt(dx**2 + dy**2)
        
        # 3. --- THE FIX: CLAMP VELOCITY ---
        # If the physics prediction tries to move the ball further than 
        # max_speed (the radius), cap it at the radius.
        if speed > self.max_speed:
            scale = self.max_speed / speed
            dx *= scale
            dy *= scale
            # shape: (we keep the direction, but reduce the distance)
        
        pred_x = int(x2 + dx)
        pred_y = int(y2 + dy)
        
        w_half = self.last_dims[0] // 2
        h_half = self.last_dims[1] // 2
        h_img, w_img = frame_shape[:2]
        
        # Strict clamping to screen boundaries
        p_x1 = max(0, min(w_img - 1, pred_x - w_half))
        p_y1 = max(0, min(h_img - 1, pred_y - h_half))
        p_x2 = max(p_x1 + 1, min(w_img, pred_x + w_half)) 
        p_y2 = max(p_y1 + 1, min(h_img, pred_y + h_half)) 
        
        self.ghost_counter += 1
        
        # We append the CLAMPED prediction to history to stabilize future frames
        self.history.append((pred_x, pred_y))
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return (p_x1, p_y1, p_x2, p_y2)

# =============================================================================
# --- COMPONENT 2: COLOR DETECTOR (Smart Fallback) ---
# =============================================================================
class ColorBlobDetector:
    def __init__(self):
        # NOTE: If camera is moving, detectShadows=False is critical
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.lower_yellow = np.array([18, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        self.warmup = 50
        self.count = 0
        self.box_padding = 0.2

    def detect(self, frame):
        self.count += 1
        fgMask = self.backSub.apply(frame)
        if self.count < self.warmup: return None

        h_img, w_img = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_and(fgMask, cv2.inRange(hsv, self.lower_yellow, self.upper_yellow))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        max_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            # Stricter area filter (ignore tiny specks)
            if area < 150 or area > 20000: continue
            
            ((cx, cy), radius) = cv2.minEnclosingCircle(c)
            if area / (np.pi * radius**2) < 0.60: continue

            x, y, w, h = cv2.boundingRect(c)
            center_x, center_y = x + w//2, y + h//2
            side = max(w, h) * (1 + self.box_padding)
            r = int(side / 2)
            
            if area > max_area:
                max_area = area
                best_box = (
                    max(0, center_x - r),
                    max(0, center_y - r),
                    min(w_img, center_x + r),
                    min(h_img, center_y + r)
                )
        return best_box

# =============================================================================
# --- MAIN PACKAGE: IVIS DETECTOR ---
# =============================================================================
class IVISDetector:
    def __init__(self, model_path, conf=0.5):
        print(f"[IVIS] Initializing. Loading YOLO from {model_path}...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[IVIS] Error loading YOLO: {e}")
            self.model = None
            
        self.conf = conf
        self.color_det = ColorBlobDetector()
        self.tracker = MomentumTracker(max_ghost_frames=2, max_speed=100)
        
        # --- NEW: Gating Threshold ---
        # If the yellow blob is more than this many pixels away from 
        # the last known ball position, ignore it.
        self.max_jump_distance = 300 
        
        self.colors = {
            'yolo': (255, 0, 0),    
            'color': (0, 255, 255), 
            'physics': (255, 0, 255)
        }

    def predict(self, frame):
        box = None
        source = None

        # 1. Try YOLO (Primary Sensor)
        if self.model:
            results = self.model.predict(frame, conf=self.conf, verbose=False, classes=[0])
            if results[0].boxes:
                b = results[0].boxes[0]
                coords = [int(i) for i in b.xyxy[0]]
                box = tuple(coords)
                source = 'yolo'

        # 2. Try Color (Secondary Sensor) WITH GATING
        if box is None:
            raw_color_box = self.color_det.detect(frame)
            
            if raw_color_box:
                # Check distance from last known position (if valid history exists)
                last_pos = self.tracker.get_last_position()
                
                if last_pos:
                    # Calculate center of the detected color blob
                    blob_cx = (raw_color_box[0] + raw_color_box[2]) / 2
                    blob_cy = (raw_color_box[1] + raw_color_box[3]) / 2
                    
                    # Calculate distance
                    dist = math.sqrt((blob_cx - last_pos[0])**2 + (blob_cy - last_pos[1])**2)
                    
                    if dist < self.max_jump_distance:
                        box = raw_color_box
                        source = 'color'
                    else:
                        # DEBUG: Uncomment to see when blobs are rejected
                        # print(f"Rejected color blob. Dist {dist:.1f} > {self.max_jump_distance}")
                        pass
                else:
                    # No history? We must accept the blob to initialize tracking
                    box = raw_color_box
                    source = 'color'

        # 3. Physics Update
        if box:
            self.tracker.update(box)
        else:
            box = self.tracker.predict(frame.shape)
            if box: source = 'physics'

        if box:
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            return {'box': box, 'source': source, 'center': (cx, cy)}
        
        return None

    def visualize(self, frame, result, mode=1):
        if result is None or mode == 4:
            return frame

        x1, y1, x2, y2 = result['box']
        color = self.colors.get(result['source'], (0, 255, 0))
        
        # --- FIX: Ensure box coordinates are valid before drawing ---
        if x2 <= x1 or y2 <= y1:
            return frame

        if mode == 1: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"IVIS:{result['source'].upper()}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        elif mode == 2: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        elif mode == 3: 
            cx, cy = result['center']
            # --- FIX: Prevent negative radius ---
            radius = max(1, max(x2-x1, y2-y1) // 2)
            
            try:
                cv2.circle(frame, (cx, cy), radius, color, 2)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            except Exception:
                pass # Ignore drawing errors if coordinates are weird

        return frame