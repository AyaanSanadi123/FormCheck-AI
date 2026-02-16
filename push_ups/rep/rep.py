import numpy as np
import time
from collections import deque

class PushUpsRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.active_side = calibration_data.get('active_side', 'RIGHT')
        self.floor_y_baseline = calibration_data.get('floor_y_baseline', 0.0)

        # Thresholds (Normalized units based on Torso Length)
        self.THRESH_DEPTH = 0.25      # Hands must come within 0.25 torso-lengths of chest (Y relative to hip=0)
        self.THRESH_LOCKOUT = 0.70    # Hands must push away to > 0.70 units (Y relative to hip=0)
        self.THRESH_HIP_DEVIATION_Y = 0.15 # Max Y-drift of ankle relative to hip (0,0) (Sag/Pike)
        self.THRESH_HEAD_DROP_Y = 0.2 # Max Y-drift of ear relative to shoulder (0,0)
        self.THRESH_ELBOW_FLARE = 60.0   # Degrees: Max angle between humerus and torso
        self.THRESH_STABILITY_X = 0.1 # Max X-drift of hip during rep
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_wrist_y = 0.0
        self.velocity = 0
        self.start_hip_x = 0.0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        sh_idx = 12 if self.active_side == 'RIGHT' else 11
        el_idx = 14 if self.active_side == 'RIGHT' else 13
        wr_idx = 16 if self.active_side == 'RIGHT' else 15
        hip_idx = 24 if self.active_side == 'RIGHT' else 23
        ank_idx = 28 if self.active_side == 'RIGHT' else 27
        ear_idx = 8 if self.active_side == 'RIGHT' else 7

        sh, el, wr, hip, ank, ear = landmarks[sh_idx], landmarks[el_idx], landmarks[wr_idx], \
                                     landmarks[hip_idx], landmarks[ank_idx], landmarks[ear_idx]
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        # Wrist Y is our proxy for depth. Lower Y means deeper. Hip is (0,0).
        current_wrist_y = wr.y 
        self.velocity = (current_wrist_y - self.prev_wrist_y) / dt # Positive = descending

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        # Continuous Form Monitoring (The "Plank" Check)
        self._check_plank_integrity(ank, ear, sh, el)

        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            if self.velocity > 0.4 and current_wrist_y < self.floor_y_baseline - 0.1: # Significant downward motion from starting plank
                self._start_rep()
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Lower your chest"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control down..."
            if current_wrist_y > self.THRESH_DEPTH: # Deeper than threshold (remember positive Y is down)
                self.state = "BOTTOM"
                self.feedback_buffer = "Push back up!"

        elif self.state == "BOTTOM":
            self.feedback_buffer = "Push!"
            if self.velocity < -0.3: # Moving up
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Push up!"
            if current_wrist_y < self.THRESH_LOCKOUT: # Returned to top position (Y is negative towards top)
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        self.prev_wrist_y = current_wrist_y

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"depth": current_wrist_y, "velocity": self.velocity}
        }

    # --- HELPERS ---

    def _check_plank_integrity(self, ank, ear, sh, el):
        """Monitors spinal alignment and elbow path."""
        # A. Hip Sag / Pike (Ankle Y relative to Hip=0)
        if ank.y > self.THRESH_HIP_DEVIATION_Y:
            self._add_fault("HIP_SAG", 15, "Keep your core tight - don't let hips sag")
        elif ank.y < -self.THRESH_HIP_DEVIATION_Y:
            self._add_fault("HIP_PIKE", 10, "Keep your hips down")

        # B. Head Position (Ear Y relative to Shoulder Y which is close to 0)
        if ear.y > self.THRESH_HEAD_DROP_Y:
            self._add_fault("HEAD_DROP", 5, "Look slightly forward, pack your neck")

        # C. Elbow Flare (Angle between Shoulder-Elbow-Wrist)
        elbow_angle = self._calculate_angle(sh, el, wr)
        if elbow_angle < 90 - (self.THRESH_ELBOW_FLARE / 2) or elbow_angle > 90 + (self.THRESH_ELBOW_FLARE / 2):
            self._add_fault("ELBOW_FLARE", 10, "Tuck your elbows in closer to your body")

    def _calculate_angle(self, p1, p2, p3):
        """Standard 2D angle math (p1-p2-p3)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))) if norm != 0 else 0

    def _start_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()
        self.start_hip_x = 0.0 # Reset for stability checks

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed - Check Plank"
