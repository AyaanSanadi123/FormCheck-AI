import numpy as np
import time

class DeadliftRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper/Normalizer)
        # Note: Normalizer ensures Floor Y = 0.
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.torso_length = calibration_data.get('torso_length', 1.0)
        
        # Thresholds (Normalized Units)
        self.THRESH_LIFTOFF = 0.05   # Bar must be 5% up to start
        self.THRESH_DRIFT = 0.15     # Max horizontal drift allowed
        self.THRESH_LOCKOUT_ANGLE = 165.0  # Min angle for full extension
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Drive with Legs"
        
        # Physics Tracking
        self.prev_bar_y = 0
        self.prev_time = 0
        self.velocity = 0
        self.max_bar_y = 0      # Track highest point (Lockout height)

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Main Logic Pipeline.
        Args:
            landmarks: Normalized landmarks (Y=0 is Floor, +Y is Up).
            raw_landmarks: Passed through for visualizer.
            timestamp: Float time in seconds.
        """
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized & Active) ---
        if self.active_side == "LEFT":
            sh = landmarks[11]; hip = landmarks[23]; knee = landmarks[25]; ankle = landmarks[27]; wrist = landmarks[15]
        else:
            sh = landmarks[12]; hip = landmarks[24]; knee = landmarks[26]; ankle = landmarks[28]; wrist = landmarks[16]
        
        # --- STEP 2: CALCULATE METRICS ---
        
        # A. Bar Height (Normalized Y)
        # Normalizer sets Floor=0. So wrist.y is height.
        bar_height = wrist.y 
        
        # B. Velocity
        curr_time = timestamp if timestamp else time.time()
        if self.prev_time == 0:
            dt = 1.0 / self.FPS
        else:
            dt = curr_time - self.prev_time
        
        if dt <= 0: dt = 0.001

        self.velocity = (bar_height - self.prev_bar_y) / dt
        
        # Update History
        self.prev_bar_y = bar_height
        self.prev_time = curr_time

        # C. Joint Angles
        hip_angle = self._calculate_angle(sh, hip, knee)
        knee_angle = self._calculate_angle(hip, knee, ankle)
        
        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (Waiting for Pull)
        if self.state == "IDLE":
            self.feedback_buffer = "Drive with Legs"
            
            # TRIGGER: Bar rises above floor threshold
            if bar_height > self.THRESH_LIFTOFF:
                self._start_rep(hip.y, sh.y)
                self.state = "ASCENDING"

        # B. ASCENDING (The Pull)
        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push Floor Away!"
            
            # TRACK: Max Height
            if bar_height > self.max_bar_y:
                self.max_bar_y = bar_height

            # FAULT 1: Bar Drift (Horizontal)
            # Normalizer sets Ankle X = 0. If Wrist X > Threshold, it's drifting forward.
            if abs(wrist.x) > self.THRESH_DRIFT:
                self._add_fault("BAR_DRIFT", 5, "Keep Bar Close!")

            # FAULT 2: Stripper Pull (Hips rising too fast)
            # Only check in first pull (Height < 40% of Torso)
            # Logic: If Hips rise X amount, Shoulders MUST rise X amount.
            if bar_height < (self.torso_length * 0.4):
                delta_hip = hip.y - self.start_hip_y
                delta_sh = sh.y - self.start_sh_y
                
                # If Hips moved up significantly more than Shoulders
                # (e.g., Hips up 20cm, Shoulders up 5cm -> Back is flattening)
                if delta_hip > (delta_sh + 0.1): 
                    self._add_fault("STRIPPER_PULL", 10, "Chest Up! Don't shoot hips.")

            # TRANSITION: Velocity drops near top OR Height is sufficient
            # We look for "Lockout Zone" (e.g. > 70% of Torso Length)
            if self.velocity < 0.1 and bar_height > (self.torso_length * 0.7):
                self.state = "LOCKOUT"

        # C. LOCKOUT (The Top)
        elif self.state == "LOCKOUT":
            self.feedback_buffer = "Hold..."
            
            # FAULT 3: Soft Hips (Severe)
            if hip_angle < self.THRESH_LOCKOUT_ANGLE:
                self._add_fault("SOFT_HIPS", 10, "Squeeze Glutes!")
                
            # FAULT 4: Soft Knees (Minor)
            if knee_angle < 170:
                self._add_fault("SOFT_KNEES", 5, "Lock Knees!")
                
            # FAULT 5: Over-Extension (Leaning back too far)
            # Check if Shoulder X is behind Hip X (negative X relative to hip)
            if sh.x < (hip.x - 0.15):
                self._add_fault("OVER_EXTEND", 5, "Don't Lean Back!")

            # TRANSITION: Bar starts going down
            if self.velocity < -0.1:
                self.state = "DESCENDING"

        # D. DESCENDING (Return to Floor)
        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control Down"
            
            # TRANSITION: Bar hits floor
            if bar_height < self.THRESH_LIFTOFF:
                self.state = "COMPLETE"

        # E. COMPLETE
        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "coords": landmarks,          
            "raw_coords": raw_landmarks,  
            "velocity": self.velocity,
            "faults": list(set([f['code'] for f in self.faults])),
            
            # Visualization Helpers
            "bar_height": bar_height,
            "hip_angle": int(hip_angle)
        }

    # --- HELPERS ---

    def _start_rep(self, current_hip_y, current_sh_y):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_bar_y = 0
        self.start_hip_y = current_hip_y
        self.start_sh_y = current_sh_y

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        # Only count if score > 50
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
