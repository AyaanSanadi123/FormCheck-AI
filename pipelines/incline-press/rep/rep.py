import numpy as np
import time
from collections import deque

class InclinePressRep:
    """
    Rep Logic for Incline Barbell/Dumbbell Press.
    Complies with PIPELINE_BLUEPRINT.md standards.
    """
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.setup_torso_angle = calibration_data.get('setup_torso_angle', 45.0)
        
        # State Management (Down-First)
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Lock arms out to start."
        
        # Hysteresis (Prevent state flickering)
        self.transition_counter = 0
        self.TRANSITION_THRESH = 3 # Consecutive frames needed
        
        # Physics Tracking
        self.prev_angle = None 
        self.prev_time = None
        self.angular_velocity = 0.0
        self.velocity_history = deque(maxlen=5) 
        
        # Rep Bounds
        self.min_elbow_angle = 180.0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT ACTIVE JOINTS ---
        if self.active_side == "LEFT":
            sh = landmarks[11]; hip = landmarks[23]
            el = landmarks[13]; wr = landmarks[15]
        else:
            sh = landmarks[12]; hip = landmarks[24]
            el = landmarks[14]; wr = landmarks[16]

        # --- STEP 2: CALCULATE METRICS ---
        curr_time = timestamp if timestamp else time.time()
        current_angle = self._calculate_angle(sh, el, wr)
        
        # Torso Angle: Shoulder relative to Hip
        # Shoulder is (0,0) in normalized space. 
        dx_torso = sh.x - hip.x
        dy_torso = sh.y - hip.y
        if dx_torso == 0: dx_torso = 0.001
        current_torso_angle = np.degrees(np.arctan2(dy_torso, dx_torso))

        # Velocity tracking
        if self.prev_angle is None:
            self.prev_angle = current_angle
            self.prev_time = curr_time
            return None 

        dt = curr_time - self.prev_time if (curr_time > self.prev_time) else 1.0/self.FPS
        raw_vel = (current_angle - self.prev_angle) / dt
        self.velocity_history.append(raw_vel)
        self.angular_velocity = np.mean(self.velocity_history)
        
        self.prev_angle = current_angle
        self.prev_time = curr_time

        # Joint Stacking Math (Horizontal drift)
        stacking_drift = abs(wr.x - el.x)

        # --- STEP 3: STATE MACHINE (Down-First) ---

        if self.state == "IDLE":
            self.feedback_buffer = "Lower weight to chest"
            if self.angular_velocity < -12.0 and current_angle < 165.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._start_rep(current_angle)
                    self.state = "ECCENTRIC"
                    self.transition_counter = 0
            else:
                self.transition_counter = 0

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the drop"
            if current_angle < self.min_elbow_angle:
                self.min_elbow_angle = current_angle

            self._check_ongoing_faults(stacking_drift, current_torso_angle)

            if self.angular_velocity > 5.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    if self.min_elbow_angle > 95.0:
                        self._add_fault("SHALLOW_REP", 5, "Go deeper!")
                    self.state = "CONCENTRIC"
                    self.transition_counter = 0
            else:
                self.transition_counter = 0

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Push!"
            self._check_ongoing_faults(stacking_drift, current_torso_angle)
            
            if current_angle >= 155.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._finish_rep(success=True)
                    self.state = "IDLE"
                    self.transition_counter = 0
            elif self.angular_velocity < -15.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._add_fault("SHORT_LOCKOUT", 5, "Lockout completely!")
                    self._finish_rep(success=False) 
                    self.state = "IDLE"
                    self.transition_counter = 0
            else:
                self.transition_counter = 0

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
                "torso_angle": int(current_torso_angle),
                "stacking_drift": round(stacking_drift, 3),
                "active_side": self.active_side
            }
        }

    # --- HELPERS ---
    def _start_rep(self, start_ang):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.min_elbow_angle = start_ang

    def _check_ongoing_faults(self, stacking_drift, current_torso_angle):
        # SEVERE FAULT 1: Unstacked Joints
        # 0.15 represents drifting 15% of torso length away from vertical alignment
        if stacking_drift > 0.15:
            self._add_fault("UNSTACKED_JOINTS", 10, "Keep your wrists directly above elbows!")
            
        # SEVERE FAULT 2: Hip Bridge
        # If live angle drops significantly below setup angle, hips are raised
        if current_torso_angle < self.setup_torso_angle - 15.0:
            self._add_fault("HIP_BRIDGE", 10, "Keep your hips on the bench!")

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