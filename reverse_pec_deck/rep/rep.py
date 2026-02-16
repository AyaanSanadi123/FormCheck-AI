import numpy as np
import time

class ReversePecDeckRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)

        # Thresholds
        self.THRESH_START_X = 0.35
        self.THRESH_TOP_X = 1.1
        self.THRESH_ELBOW_CHANGE = 15.0
        self.THRESH_SHRUG = -0.15
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_wrist_dist = 0.0
        self.velocity = 0
        self.initial_elbow_angle = 160.0
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        l_wr, r_wr = landmarks[15], landmarks[16]
        l_el, r_el = landmarks[13], landmarks[14]
        l_sh, r_sh = landmarks[11], landmarks[12]
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        
        current_wrist_dist = abs(l_wr.x - r_wr.x)
        self.velocity = (current_wrist_dist - self.prev_wrist_dist) / dt

        l_angle = self._calc_angle(l_sh, l_el, l_wr)
        r_angle = self._calc_angle(r_sh, r_el, r_wr)
        avg_angle = (l_angle + r_angle) / 2

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            if self.velocity > 0.5 and current_wrist_dist > self.THRESH_START_X:
                self._start_rep()
                self.initial_elbow_angle = avg_angle
                self.state = "CONCENTRIC"
                self.feedback_buffer = "Sweep out wide"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze!"

            if l_sh.y < self.THRESH_SHRUG or r_sh.y < self.THRESH_SHRUG:
                self._add_fault("SHOULDER_SHRUG", 15, "Keep shoulders down")
            
            if abs(avg_angle - self.initial_elbow_angle) > self.THRESH_ELBOW_CHANGE:
                self._add_fault("ELBOW_PIVOT", 20, "Keep elbows locked")

            if current_wrist_dist > self.THRESH_TOP_X:
                self.state = "TOP"
                self.feedback_buffer = "Hold the squeeze!"

        elif self.state == "TOP":
            self.feedback_buffer = "Hold!"
            if self.velocity < -0.3:
                self.state = "ECCENTRIC"
                self.feedback_buffer = "Control the return"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the release..."
            if current_wrist_dist < self.THRESH_START_X:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        self.prev_wrist_dist = current_wrist_dist

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"expansion": current_wrist_dist, "elbow_angle": avg_angle}
        }
        
    def _calc_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))) if norm != 0 else 0

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
        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"
