import numpy as np
import time

class PecDeckRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.shoulder_y_baseline = calibration_data.get('shoulder_y_baseline', 0.5)

        # Thresholds (Normalized Units based on Shoulder Width)
        self.THRESH_PEAK_SQUEEZE = 0.35  # Max X-dist between wrists for valid rep
        self.THRESH_ECCENTRIC_STOP = 0.9 # Min X-dist to reset to IDLE
        self.THRESH_SHOULDER_HIKE = 0.15 # Max Y-drift of shoulders (Shrugging)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_dist = 1.0
        self.velocity = 0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        l_wrist, r_wrist = landmarks[15], landmarks[16]
        l_sh, r_sh = landmarks[11], landmarks[12]
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        # Calculate Horizontal Adduction Distance (Wrist to Midline)
        current_dist = abs(l_wrist.x - r_wrist.x)
        self.velocity = (self.prev_dist - current_dist) / dt  # Positive = Closing

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            if current_dist < 0.8 and self.velocity > 0.5:
                self._start_rep()
                self.state = "CONCENTRIC"
                self.feedback_buffer = "Squeeze together!"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze!"
            
            # FAULT: Shoulder Hiking
            if (l_sh.y < -self.THRESH_SHOULDER_HIKE) or (r_sh.y < -self.THRESH_SHOULDER_HIKE):
                self._add_fault("SHOULDER_HIKE", 15, "Keep shoulders down")

            if current_dist < self.THRESH_PEAK_SQUEEZE:
                self.state = "TOP"
                self.feedback_buffer = "Hold!"

        elif self.state == "TOP":
            self.feedback_buffer = "Hold the squeeze!"
            if self.velocity < -0.3:
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Open slowly"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the release..."
            if current_dist > self.THRESH_ECCENTRIC_STOP:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        # Update trackers
        self.prev_dist = current_dist

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"wrist_dist": current_dist, "velocity": self.velocity}
        }

    # --- HELPERS ---

    def _start_rep(self):
        """Reset score and faults for new rep."""
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per rep."""
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        """Finalizes the rep."""
        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"
