import numpy as np
import time

class HangingLegRaisesRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.active_side = calibration_data.get('active_side', 'RIGHT')

        # Thresholds (Normalized units)
        self.THRESH_TOP_ANGLE = 100.0  # Torso-to-Femur angle (degrees)
        self.THRESH_SWING_X = 0.25     # Max Hip X-deviation (Kipping)
        self.THRESH_KNEE_BEND = 160.0  # Minimum knee extension for 'strict'
        self.THRESH_RESET_VEL = 0.05   # Velocity threshold for 'Dead Stop'
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Wait for a dead hang..."
        
        # Physics Tracking
        self.prev_ank_y = 0.0
        self.velocity = 0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        sh_idx = 12 if self.active_side == 'RIGHT' else 11
        hip_idx = 24 if self.active_side == 'RIGHT' else 23
        knee_idx = 26 if self.active_side == 'RIGHT' else 25
        ank_idx = 28 if self.active_side == 'RIGHT' else 27

        sh, hip, knee, ank = landmarks[sh_idx], landmarks[hip_idx], landmarks[knee_idx], landmarks[ank_idx]
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        # Angular Velocity of the leg (Ankle Y change)
        self.velocity = (ank.y - self.prev_ank_y) / dt
        
        # Calculate Core Metrics
        torso_femur_angle = self._get_angle(sh, hip, knee)
        knee_angle = self._get_angle(hip, knee, ank)

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        if self.state == "IDLE":
            self.feedback_buffer = "Wait for a dead hang..."
            if abs(ank.x) < 0.1 and abs(self.velocity) < self.THRESH_RESET_VEL:
                self._start_rep()
                self.state = "CONCENTRIC"
                self.feedback_buffer = "Lift with your core"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Lift!"
            self._check_faults(hip, knee_angle)
            
            if torso_femur_angle < self.THRESH_TOP_ANGLE:
                self.state = "TOP"
                self.feedback_buffer = "Great height! Lower slowly."

        elif self.state == "TOP":
            self.feedback_buffer = "Hold!"
            if self.velocity > 0.1: # Legs starting to descend
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Control the descent"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the descent..."
            if self.velocity > 2.5:
                self._add_fault("FAST_DROP", 10, "Control the descent")
            
            if ank.y > (hip.y + 0.5):
                self.state = "RESET"
                self.feedback_buffer = "Stop the swing before next rep"

        elif self.state == "RESET":
            self.feedback_buffer = "Stop the swing before next rep"
            if abs(self.velocity) < self.THRESH_RESET_VEL and abs(ank.x) < 0.1:
                self._finish_rep()
                self.state = "IDLE"

        self.prev_ank_y = ank.y

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"angle": torso_femur_angle, "swing": hip.x}
        }

    # --- HELPERS ---

    def _get_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        norm = (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0))) if norm != 0 else 0

    def _check_faults(self, hip, knee_angle):
        if abs(hip.x) > self.THRESH_SWING_X:
            self._add_fault("SWINGING", 20, "Keep your torso still")
            
        if knee_angle < self.THRESH_KNEE_BEND:
            self._add_fault("KNEE_BEND", 15, "Keep your legs straight")

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _start_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()

    def _finish_rep(self):
        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"
