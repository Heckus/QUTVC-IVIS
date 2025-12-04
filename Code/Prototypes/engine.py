import cv2
import numpy as np
from sklearn.cluster import KMeans

# =============================================================================
# --- COMPONENT 1: GEOMETRY & PHYSICS HELPERS ---
# =============================================================================

class NetHomography:
    """ Maps 2D pixel coordinates of the net to Real-World Meters. """
    def __init__(self):
        # Standard Net: 9m wide, 2.43m top, 1.43m bottom (approx 1m mesh)
        self.real_world_points = np.array([
            [0, 2.43], [9.0, 2.43], [9.0, 1.43], [0, 1.43]
        ], dtype=np.float32)
        self.matrix = None
        self.net_bottom_line = None 

    def compute_matrix(self, pixel_points):
        if len(pixel_points) != 4: return False
        pts_src = np.array(pixel_points, dtype=np.float32)
        
        self.matrix, _ = cv2.findHomography(pts_src, self.real_world_points)
        
        # Compute "Bottom Line" for Player Filtering
        p_bl, p_br = pts_src[3], pts_src[2]
        if p_br[0] - p_bl[0] != 0:
            slope = (p_br[1] - p_bl[1]) / (p_br[0] - p_bl[0])
            intercept = p_bl[1] - slope * p_bl[0]
            self.net_bottom_line = (slope, intercept)
        else:
            self.net_bottom_line = None 
        return True

    def get_real_height(self, x, y):
        if self.matrix is None: return 0.0
        point = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(point, self.matrix)
        return max(0.0, dst[0][0][1])

    def is_player_on_close_side(self, foot_x, foot_y):
        if self.net_bottom_line is None: return True
        m, c = self.net_bottom_line
        line_y = m * foot_x + c
        # +10 buffer: Player must be clearly below the line
        return foot_y > (line_y + 10) 

class BallKalmanFilter:
    """ Used purely for Analytics (Velocity Calculation) """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0], [0, 1, 0, 1],
            [0, 0, 1, 0], [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01 
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.initialized = False

    def update(self, cx, cy):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        return self.kf.predict()

    def get_velocity(self):
        if not self.initialized: return (0, 0)
        return (self.kf.statePost[2][0], self.kf.statePost[3][0])

# =============================================================================
# --- COMPONENT 2: PERSISTENT PLAYER TRACKER (UPDATED) ---
# =============================================================================

class PersistentPlayerTracker:
    """
    Manages exactly 6 players. 
    Uses Momentum (Velocity) matching to handle fast movements.
    """
    def __init__(self):
        # { 'P1': {
        #     'box': [x1,y1,x2,y2], 
        #     'kps': [...], 
        #     'center': (x,y),
        #     'velocity': (vx, vy), 
        #     'missing_frames': 0, 
        #     'color': (R,G,B)
        #   } 
        # }
        self.tracks = {}
        self.is_initialized = False
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (0, 255, 255), (255, 0, 255)
        ]

    def register_player(self, label, box):
        """ Register a player at start (Human in the Loop) """
        color_idx = int(label.replace('P', '')) - 1
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        self.tracks[label] = {
            'box': box,
            'kps': None,
            'center': (cx, cy),
            'velocity': (0, 0), # New: Track speed
            'missing_frames': 0,
            'color': self.colors[color_idx % 6]
        }
        self.is_initialized = True

    def update(self, detections):
        """
        Matches detections to known players using Predicted Position (Center + Velocity).
        detections: list of {'box': np.array, 'kps': np.array}
        """
        if not self.is_initialized or not detections:
            for label in self.tracks:
                self.tracks[label]['missing_frames'] += 1
            return self.tracks

        # 1. Predict Next Positions
        predicted_positions = {}
        for label, data in self.tracks.items():
            cx, cy = data['center']
            vx, vy = data['velocity']
            # Simple linear prediction
            pred_x = cx + vx
            pred_y = cy + vy
            predicted_positions[label] = (pred_x, pred_y)

        # 2. Create Cost Matrix (Distance between Prediction and Detection)
        labels = list(self.tracks.keys())
        cost_matrix = np.zeros((len(labels), len(detections)))
        
        for i, label in enumerate(labels):
            pred_cx, pred_cy = predicted_positions[label]
            for j, det in enumerate(detections):
                d_box = det['box']
                d_cx = (d_box[0] + d_box[2]) / 2
                d_cy = (d_box[1] + d_box[3]) / 2
                dist = np.hypot(pred_cx - d_cx, pred_cy - d_cy)
                cost_matrix[i, j] = dist

        # 3. Greedy Assignment
        assignments = {} 
        assigned_dets = set()
        MAX_DIST = 250 # Increased slightly for momentum tolerance

        while len(assignments) < len(labels) and len(assigned_dets) < len(detections):
            min_val = np.min(cost_matrix)
            if min_val > MAX_DIST: break 
            
            r, c = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            if cost_matrix[r, c] == np.inf: break
            
            assignments[r] = c
            assigned_dets.add(c)
            cost_matrix[r, :] = np.inf
            cost_matrix[:, c] = np.inf

        # 4. Update State
        alpha = 0.7 # Smoothing factor for velocity
        
        for i, label in enumerate(labels):
            if i in assignments:
                det_idx = assignments[i]
                new_box = detections[det_idx]['box']
                new_kps = detections[det_idx]['kps']
                
                # Calculate new center
                new_cx = (new_box[0] + new_box[2]) / 2
                new_cy = (new_box[1] + new_box[3]) / 2
                
                # Calculate instant velocity
                old_cx, old_cy = self.tracks[label]['center']
                inst_vx = new_cx - old_cx
                inst_vy = new_cy - old_cy
                
                # Smooth velocity
                old_vx, old_vy = self.tracks[label]['velocity']
                smooth_vx = alpha * inst_vx + (1 - alpha) * old_vx
                smooth_vy = alpha * inst_vy + (1 - alpha) * old_vy
                
                self.tracks[label].update({
                    'box': new_box,
                    'kps': new_kps,
                    'center': (new_cx, new_cy),
                    'velocity': (smooth_vx, smooth_vy),
                    'missing_frames': 0
                })
            else:
                self.tracks[label]['missing_frames'] += 1
                # Decay velocity if lost
                vx, vy = self.tracks[label]['velocity']
                self.tracks[label]['velocity'] = (vx * 0.9, vy * 0.9)
        
        return self.tracks

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)