import numpy as np
import time

class RdlsRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.base_knee_angle = calibration_data.get('base_knee_angle', 170.0)
        self.active_side = calibration_data.get('active_side', 'RIGHT')
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Stand tall to begin"
        
        # Thresholds (Normalized Units / Degrees)
        self.THRESH_HINGE_START_X = -0.05 # Hip X movement to start eccentric
        self.THRESH_DEPTH_X = -0.20    # Target Hip X for 'Stretch' state
        self.THRESH_KNEE_VAR = 8.0      # Max allowed change in knee angle from baseline
        self.THRESH_BAR_DRIFT = 0.15    # Max Wrist-to-Ankle X distance
        self.THRESH_LUMBAR = 165.0      # Min Sh-Hip-Knee angle (linearity)

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks: return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        sh_idx = 12 if self.active_side == 'RIGHT' else 11
        hip_idx = 24 if self.active_side == 'RIGHT' else 23
        knee_idx = 26 if self.active_side == 'RIGHT' else 25
        ank_idx = 28 if self.active_side == 'RIGHT' else 27
        wr_idx = 16 if self.active_side == 'RIGHT' else 15

        sh, hip, knee, ank, wr = landmarks[sh_idx], landmarks[hip_idx], landmarks[knee_idx], landmarks[ank_idx], landmarks[wr_idx]
        
        # --- STEP 2: CALCULATE METRICS ---
        curr_knee_angle = self._calculate_angle(hip, knee, ank)
        spine_angle = self._calculate_angle(sh, hip, knee)
        hip_travel_x = hip.x # Hips move in -X direction
        bar_distance_x = abs(wr.x - ank.x) # Ankle X is 0 in normalized space

        # --- STEP 3: STATE MACHINE ---
        if self.state == "IDLE":
            self.feedback_buffer = "Stand tall to begin"
            if hip_travel_x < self.THRESH_HINGE_START_X: # Hips start moving back
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Hips back, feel the stretch"
                self._start_rep()

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Hinge back..."
            self._check_faults(curr_knee_angle, spine_angle, bar_distance_x)
            
            if hip_travel_x < self.THRESH_DEPTH_X:
                self.state = "STRETCH"
                self.feedback_buffer = "Peak tension. Drive hips forward."

        elif self.state == "STRETCH":
            self.feedback_buffer = "Drive hips forward!"
            if hip_travel_x > (self.THRESH_DEPTH_X + 0.05): # Hips begin returning
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze glutes at top!"
            if hip_travel_x > self.THRESH_HINGE_START_X: # Hips returned to neutral
                self._finish_rep()
                self.state = "IDLE"
                self.feedback_buffer = "Good rep. Reset and repeat."

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": int(self.current_score),
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"hip_x": hip_travel_x, "knee_angle": curr_knee_angle}
        }

    # --- HELPERS ---

    def _calculate_angle(self, p1, p2, p3):
        """Calculates angle between three points (p1-p2-p3)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))) if norm != 0 else 0

    def _check_faults(self, knee_angle, spine_angle, bar_dist):
        # A. Squat Fault (Knees bending too much)
        if abs(knee_angle - self.base_knee_angle) > self.THRESH_KNEE_VAR:
            self._add_fault("SQUAT_FAULT", 15, "Don't bend your knees—hinge at the hips")

        # B. Lumbar Rounding
        if spine_angle < self.THRESH_LUMBAR:
            self._add_fault("LUMBAR_ROUND", 20, "Keep your back flat")

        # C. Bar Drift
        if bar_dist > self.THRESH_BAR_DRIFT:
            self._add_fault("BAR_DRIFT", 10, "Keep the weight close to your shins")

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
        if self.current_score > 60: # Quality threshold
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Discounted - Focus on Form"
