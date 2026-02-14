import numpy as np
import time

class BenchPressRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.arm_length = calibration_data.get('arm_length', 1.0) # Scale factor
        
        # Thresholds (Normalized to Arm Length)
        self.THRESH_ROM = 0.90       
        self.THRESH_BRIDGE = 0.15    # Max hip lift allowed
        self.THRESH_BOUNCE = 3.5     
        self.THRESH_FLARE = 85.0     
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Unrack Bar"
        
        # Physics Tracking
        self.prev_bar_y = 0.0
        self.velocity = 0.0
        self.prev_velocity = 0.0 
        self.min_bar_y = 1000.0  # Track lowest point (closest to chest)
        self.start_x = 0.0       
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized & Active) ---
        if self.active_side == "LEFT":
            sh = landmarks[11]; el = landmarks[13]; wr = landmarks[15]; hip = landmarks[23]
        else:
            sh = landmarks[12]; el = landmarks[14]; wr = landmarks[16]; hip = landmarks[24]
        
        # --- STEP 2: CALCULATE METRICS ---
        elbow_angle = self._calculate_angle(sh, el, wr)
        curr_bar_y = wr.y # Normalized Y (Up is +)
        dt = 1.0 / self.FPS
        
        if self.prev_bar_y != 0:
            self.velocity = (curr_bar_y - self.prev_bar_y) / dt
        self.prev_bar_y = curr_bar_y
        
        acceleration = (self.velocity - self.prev_velocity) / dt if self.prev_velocity != 0 else 0
        self.prev_velocity = self.velocity

        # Asymmetry check (needs both wrists)
        if raw_landmarks:
            l_wr_y = getattr(landmarks[15], 'y')
            r_wr_y = getattr(landmarks[16], 'y')
            if abs(l_wr_y - r_wr_y) > 0.15: # 15% of arm length tilt
                 self._add_fault("ASYMMETRY", 10, "Push Evenly!")

        # --- STEP 3: STATE MACHINE ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            # TRIGGER: Elbows unlock, Bar moves down, Aligned with shoulders
            if elbow_angle < 165 and self.velocity < -0.2: # Downward is negative in normalized space
                if abs(wr.x - sh.x) < 0.15:
                    self._start_rep(wr.x)
                    self.state = "DESCENDING"
                else:
                    self.feedback_buffer = "Set Bar Over Shoulders"

        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control Down..."
            # Track lowest point (closest to chest). Normalized Y: Up is +, Chest is 0.
            # So DESCENDING means moving from e.g. 1.0 towards 0.0.
            if curr_bar_y < self.min_bar_y: 
                 self.min_bar_y = curr_bar_y
            
            # FAULT: Glute Bridge (Hip rises significantly above bench height 0)
            if hip.y > self.THRESH_BRIDGE: 
                 self._add_fault("GLUTE_BRIDGE", 10, "Keep Hips Down!")

            # FAULT: Elbow Flare
            flare_angle = self._calculate_angle(hip, sh, el)
            if flare_angle > self.THRESH_FLARE:
                self._add_fault("ELBOW_FLARE", 5, "Tuck Elbows!")

            # TRANSITION: Velocity flips to positive (Moving Up)
            if self.velocity > 0.1:
                self.state = "ASCENDING"
                # CHECK DEPTH: Chest is at Y=0. min_bar_y should be near 0.
                if self.min_bar_y > 0.15:
                    self._add_fault("SHALLOW", 5, "Touch Your Chest!")
                if abs(acceleration) > self.THRESH_BOUNCE:
                    self._add_fault("BOUNCE", 5, "Don't Bounce!")

        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push Back!"
            if abs(wr.x - self.start_x) < 0.02:
                 self._add_fault("BAD_PATH", 5, "Push Back Towards Face")
            
            if elbow_angle > 165 and abs(self.velocity) < 0.1:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "angle": int(elbow_angle),
            "coords": landmarks,
            "raw_coords": raw_landmarks,      
            "velocity": self.velocity,
            "faults": list(set([f['code'] for f in self.faults])) 
        }

    def _start_rep(self, current_x):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.min_bar_y = 1000.0
        self.start_x = current_x

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

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))
