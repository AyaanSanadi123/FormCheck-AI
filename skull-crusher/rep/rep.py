import numpy as np
import time
from collections import deque

class SkullCrusherRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.facing_side = calibration_data.get('facing_side', 1.0)
        self.shoulder_origin_x = calibration_data.get('shoulder_origin_x', 0.5)
        self.shoulder_origin_y = calibration_data.get('shoulder_origin_y', 0.5)
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # Calculate Normalized Baseline (The user's specific starting elbow position)
        raw_base_el_x = calibration_data.get('baseline_el_x', 0.5)
        raw_base_el_y = calibration_data.get('baseline_el_y', 0.5) 
        
        if self.scale_factor < 0.001: self.scale_factor = 1.0
        
        # Normalize baseline coordinates to match the incoming frame data (Canonical Space)
        self.baseline_el_x = ((raw_base_el_x - self.shoulder_origin_x) * self.facing_side) / self.scale_factor
        self.baseline_el_y = (self.shoulder_origin_y - raw_base_el_y) / self.scale_factor
        
        # State Management (Eccentric-First)
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Lock arms out to start."
        
        # Physics Tracking
        self.prev_angle = None 
        self.prev_time = None
        self.angular_velocity = 0.0
        self.velocity_history = deque(maxlen=5) 
        
        # Rep Bounds
        self.min_drop_angle = 180.0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT ACTIVE JOINTS ---
        # Shoulder is (0,0). Head is -X, Feet are +X. Up to ceiling is +Y.
        if self.active_side == "LEFT":
            sh = landmarks[11]; el = landmarks[13]; wr = landmarks[15]
        else:
            sh = landmarks[12]; el = landmarks[14]; wr = landmarks[16]

        # --- STEP 2: CALCULATE METRICS ---
        current_angle = self._calculate_angle(sh, el, wr)
        curr_time = timestamp if timestamp else time.time()
        
        if self.prev_angle is None:
            self.prev_angle = current_angle
            self.prev_time = curr_time
            return None 

        # Smoothed Angular Velocity (Negative = dropping, Positive = pushing)
        if self.prev_time is None or curr_time == self.prev_time:
            dt = 1.0 / self.FPS # Fallback
        else:
            dt = curr_time - self.prev_time
            
        if dt <= 0: dt = 0.001
            
        raw_vel = (current_angle - self.prev_angle) / dt
        self.velocity_history.append(raw_vel)
        self.angular_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        
        self.prev_angle = current_angle
        self.prev_time = curr_time

        # Elbow Drift (Euclidean distance from baseline in "Arm Units")
        elbow_drift = np.sqrt((el.x - self.baseline_el_x)**2 + (el.y - self.baseline_el_y)**2)

        # --- STEP 3: STATE MACHINE (Eccentric-First) ---

        # A. IDLE (Arms locked out at top)
        if self.state == "IDLE":
            self.feedback_buffer = "Lower weight to forehead"
            
            # TRIGGER: Bending arms rapidly
            if self.angular_velocity < -15.0 and current_angle < 160.0:
                self._start_rep(current_angle)
                self.state = "ECCENTRIC"

        # B. ECCENTRIC (Lowering - Hands moving toward head)
        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the descent"
            
            if current_angle < self.min_drop_angle:
                self.min_drop_angle = current_angle

            # FAULT: Shoulder Swing (Breaking the anchor)
            if elbow_drift > 0.20:
                self._add_fault("SHOULDER_SWING", 10, "Keep elbows locked in space!")

            # TRANSITION: Reversing direction (Velocity flips positive)
            if self.angular_velocity > 5.0 or (current_angle > self.min_drop_angle + 10.0):
                # CHECK SHALLOW DROP
                # Good stretch is <= 90 deg.
                if self.min_drop_angle > 90.0:
                    self._add_fault("SHALLOW_DROP", 5, "Lower the bar closer to your head!")
                
                self.state = "CONCENTRIC"

        # C. CONCENTRIC (Pushing - Hands moving UP)
        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze the triceps!"
            
            # FAULT: Shoulder Swing (Using lats to heave weight)
            if elbow_drift > 0.20:
                self._add_fault("SHOULDER_SWING", 10, "Don't use your lats to push!")
            
            # TRANSITION: Arms are locked out again
            if current_angle >= 160.0:
                self._finish_rep(success=True)
                self.state = "IDLE"
                
            # EARLY REVERSAL (Short lockout / Half-repping at the bottom)
            elif self.angular_velocity < -15.0:
                self._add_fault("SHORT_LOCKOUT", 5, "Lock your arms out completely!")
                self._finish_rep(success=False) 
                self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
        # Fully compliant with PIPELINE_BLUEPRINT.md
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "coords": landmarks,          
            "raw_coords": raw_landmarks,  
            "faults": list(set([f['code'] for f in self.faults])),
            "metrics": {
                "elbow_angle": int(current_angle),
                "elbow_drift": elbow_drift,
                "active_side": self.active_side
            }
        }

    # --- HELPERS ---
    def _start_rep(self, start_ang):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.min_drop_angle = start_ang

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self, success=True):
        if success and self.current_score > 50:
            self.rep_count += 1
        elif not success:
            self.feedback_buffer = "Rep Failed (Bad Form)"

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))