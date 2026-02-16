import numpy as np
import time

class PullUpsRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.bar_y_baseline = calibration_data.get('bar_y_baseline', 0.0)

        # Thresholds (Normalized units based on Torso Length)
        self.THRESH_CHIN_OVER_BAR_Y = -0.05 # Chin Y must be above bar (Negative Y)
        self.THRESH_DEAD_HANG_Y = 0.1 # Shoulders below bar
        self.THRESH_KIP_X = 0.25        # Max horizontal drift (Kipping)
        self.THRESH_SCAPULAR_GAP = 0.1  # Min Ear-to-Shoulder distance (normalized Y)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_sh_y = 0.0
        self.velocity = 0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        chin = landmarks[0] 
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_ear, r_ear = landmarks[7], landmarks[8]
        hip = landmarks[24] 
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        current_sh_y = (l_sh.y + r_sh.y) / 2
        self.velocity = (current_sh_y - self.prev_sh_y) / dt # Negative = Moving Up

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        if self.state == "IDLE":
            self.feedback_buffer = "Dead hang"
            if self.velocity < -0.5: # User starts moving up
                self._start_rep()
                # Check for "Active Hang" - Shoulders should depress
                ear_sh_dist_left = abs(l_ear.y - l_sh.y) / self.scale_factor
                ear_sh_dist_right = abs(r_ear.y - r_sh.y) / self.scale_factor
                if ear_sh_dist_left < self.THRESH_SCAPULAR_GAP or ear_sh_dist_right < self.THRESH_SCAPULAR_GAP:
                    self._add_fault("SHRUGGED_PULL", 10, "Depress shoulders first")
                
                self.state = "CONCENTRIC"
                self.feedback_buffer = "Pull chin over bar!"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Pull!"
            # Continuous Fault: Kipping (Excessive horizontal swing)
            if abs(hip.x) > self.THRESH_KIP_X:
                self._add_fault("KIP_SWING", 20, "Stop swinging your legs")

            if chin.y < self.THRESH_CHIN_OVER_BAR_Y: # Chin Y is negative above bar
                self.state = "TOP"
                self.feedback_buffer = "Clear! Now control the descent"

        elif self.state == "TOP":
            self.feedback_buffer = "Hold!"
            if self.velocity > 0.3: # User starts descending
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Control the descent"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control down..."
            if current_sh_y > self.THRESH_DEAD_HANG_Y: # Shoulders below bar
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        self.prev_sh_y = current_sh_y

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"shoulder_y": current_sh_y, "velocity": self.velocity}
        }

    # --- HELPERS ---

    def _start_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        if self.current_score > 40:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Discounted - Focus on Form"
