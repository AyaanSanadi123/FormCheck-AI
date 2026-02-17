import numpy as np
import time
from collections import deque

class RepLogic:
    """
    Rep Logic for Cable Tricep Kickback.
    Complies with PIPELINE_BLUEPRINT.md standards.
    """
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.setup_torso_angle = calibration_data.get('setup_torso_angle', 45.0)
        
        # Normalize the Baseline Elbow so it matches incoming frame data
        facing_side = calibration_data.get('facing_side', 1.0)
        shoulder_origin_x = calibration_data.get('shoulder_origin_x', 0.5)
        shoulder_origin_y = calibration_data.get('shoulder_origin_y', 0.5)
        scale_factor = calibration_data.get('scale_factor', 1.0)
        raw_base_el_x = calibration_data.get('baseline_el_x', 0.5)
        raw_base_el_y = calibration_data.get('baseline_el_y', 0.5)
        
        if scale_factor < 0.001: scale_factor = 1.0
        
        self.baseline_el_x = ((raw_base_el_x - shoulder_origin_x) * facing_side) / scale_factor
        self.baseline_el_y = (shoulder_origin_y - raw_base_el_y) / scale_factor
        
        # State Management (Concentric-First)
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Kick the weight back."
        
        # Hysteresis (Prevent state flickering)
        self.transition_counter = 0
        self.TRANSITION_THRESH = 3 
        
        # Physics Tracking
        self.prev_angle = None 
        self.prev_time = None
        self.angular_velocity = 0.0
        self.velocity_history = deque(maxlen=5) 
        
        # Rep Bounds
        self.max_elbow_angle = 0.0
        self.last_rep_time = 0

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
        
        # Torso Angle: Live Check for Bobbing
        dx_torso = sh.x - hip.x
        dy_torso = sh.y - hip.y
        if dx_torso == 0: dx_torso = 0.001
        current_torso_angle = np.degrees(np.arctan2(abs(dy_torso), abs(dx_torso)))

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

        # Elbow Drift (Euclidean distance from normalized baseline in "Arm Units")
        elbow_drift = np.sqrt((el.x - self.baseline_el_x)**2 + (el.y - self.baseline_el_y)**2)

        # --- STEP 3: STATE MACHINE (Concentric-First) ---

        if self.state == "IDLE":
            # Only show default prompt if enough time has passed since last rep
            if curr_time - self.last_rep_time > 2.0:
                self.feedback_buffer = "Extend your arm"
                
            if self.angular_velocity > 12.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._start_rep(current_angle)
                    self.state = "CONCENTRIC"
                    self.transition_counter = 0
            else:
                self.transition_counter = 0

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze the triceps!"
            if current_angle > self.max_elbow_angle:
                self.max_elbow_angle = current_angle

            self._check_ongoing_faults(elbow_drift, current_torso_angle)

            # Reversing direction (Velocity flips negative)
            if self.angular_velocity < -5.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    if self.max_elbow_angle < 160.0:
                        self._add_fault("SHORT_LOCKOUT", 5, "Straighten your arm completely!")
                    self.state = "ECCENTRIC"
                    self.transition_counter = 0
            else:
                self.transition_counter = 0

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the return"
            self._check_ongoing_faults(elbow_drift, current_torso_angle)
            
            # Arm is flexed again
            if current_angle <= 110.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._finish_rep(success=True, timestamp=curr_time)
                    self.state = "IDLE"
                    self.transition_counter = 0
            # Early Reversal (bouncing out of the bottom)
            elif self.angular_velocity > 15.0:
                self.transition_counter += 1
                if self.transition_counter >= self.TRANSITION_THRESH:
                    self._finish_rep(success=False, timestamp=curr_time) 
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
                "elbow_drift": round(elbow_drift, 3),
                "active_side": self.active_side
            }
        }

    # --- HELPERS ---
    def _start_rep(self, start_ang):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_elbow_angle = start_ang

    def _check_ongoing_faults(self, elbow_drift, current_torso_angle):
        # SEVERE FAULT 1: Pendulum Swing (-10)
        if elbow_drift > 0.20:
            self._add_fault("PENDULUM_SWING", 10, "Keep your elbow pinned!")
            
        # SEVERE FAULT 2: Torso Bob (-10)
        if abs(current_torso_angle - self.setup_torso_angle) > 15.0:
            self._add_fault("TORSO_BOB", 10, "Keep your back still!")

    def _add_fault(self, code, penalty, msg):
        # Check if fault already recorded for this rep
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        
        # Priority-based feedback: Severe faults override minor ones. 
        # If multiple of same severity, latest one wins or we can concatenate.
        # For now, we update feedback if it's the first fault or a high-penalty one.
        if len(self.faults) == 1 or penalty >= 10:
            self.feedback_buffer = msg 

    def _finish_rep(self, success=True, timestamp=None):
        self.last_rep_time = timestamp if timestamp else time.time()
        if success:
            if self.current_score >= 70:
                self.rep_count += 1
                self.feedback_buffer = f"Good Rep! Score: {self.current_score}"
            elif self.current_score > 50:
                self.rep_count += 1
                self.feedback_buffer = f"Rep Counted (Watch Form)"
            else:
                # Rep completed but form was too poor to count
                self.feedback_buffer = "Rep Discarded: Improve Form"
        else:
            self.feedback_buffer = "Rep Failed"

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))