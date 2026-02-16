import numpy as np
import time

class LungeRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.active_side = calibration_data.get('active_side', 'RIGHT')

        # Thresholds (Normalized units based on Torso Length)
        self.THRESH_DEPTH = -0.15      # Back knee must get close to floor (Y=0)
        self.THRESH_KNEE_SHEAR = 0.25  # Front knee X-offset relative to ankle (0,0)
        self.THRESH_TORSO_LEAN = 25.0  # Max degrees lean from vertical
        self.THRESH_STABILITY_X = 0.15 # Max horizontal hip drift during rep
        self.THRESH_RETURN_HEIGHT = -0.8 # Hip Y to consider rep complete

        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.initial_hip_x = 0.0
        self.prev_hip_y = -1.0
        self.velocity = 0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        sh_idx = 12 if self.active_side == 'RIGHT' else 11
        f_hip_idx = 24 if self.active_side == 'RIGHT' else 23
        f_knee_idx = 26 if self.active_side == 'RIGHT' else 25
        b_knee_idx = 25 if self.active_side == 'RIGHT' else 26
        
        f_hip, f_knee = landmarks[f_hip_idx], landmarks[f_knee_idx]
        b_knee = landmarks[b_knee_idx]
        sh = landmarks[sh_idx]
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        self.velocity = (f_hip.y - self.prev_hip_y) / dt

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            if self.velocity > 0.3: # User starts dropping
                self._start_rep()
                self.initial_hip_x = f_hip.x
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Drop straight down"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control down..."
            self._check_lunge_geometry(f_knee, f_hip, sh, b_knee)
            
            if b_knee.y > self.THRESH_DEPTH:
                self.state = "BOTTOM"
                self.feedback_buffer = "Drive back up!"

        elif self.state == "BOTTOM":
            self.feedback_buffer = "Drive up!"
            if self.velocity < -0.2:
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self._check_lunge_geometry(f_knee, f_hip, sh, b_knee)

            if abs(f_hip.x - self.initial_hip_x) > self.THRESH_STABILITY_X:
                self._add_fault("HIP_DRIFT", 15, "Stop swaying forward/backward")

            if f_hip.y < self.THRESH_RETURN_HEIGHT: # Returned to standing height
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        self.prev_hip_y = f_hip.y

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"hip_y": f_hip.y, "knee_shear": f_knee.x}
        }

    # --- HELPERS ---

    def _check_lunge_geometry(self, f_knee, f_hip, sh, b_knee):
        # A. Knee Shear: Front knee drifting past toes (X > 0)
        if f_knee.x > self.THRESH_KNEE_SHEAR:
            self._add_fault("KNEE_SHEAR", 20, "Keep front knee over your ankle")

        # B. Torso Lean: Angle between Shoulder and Hip
        dx, dy = sh.x - f_hip.x, sh.y - f_hip.y
        lean_angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        if lean_angle > self.THRESH_TORSO_LEAN:
            self._add_fault("TORSO_LEAN", 10, "Keep your chest up")

        # C. Valgus Check (Z-axis wobble)
        if abs(f_knee.z) > 0.2:
            self._add_fault("KNEE_WOBBLE", 15, "Stabilize your front knee")

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
