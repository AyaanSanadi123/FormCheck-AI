import numpy as np
import time

class OverheadPressRepCounter:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.shoulder_y = calibration_data.get('shoulder_y', 0)     # Shoulder height (Standing)
        self.arm_length = calibration_data.get('arm_length', 1.0) # Scale factor
        self.facing_side = calibration_data.get('facing_side', 1) # 1=Right, -1=Left
        
        # Thresholds (Normalized to Arm Length)
        self.THRESH_ROM = 0.90       # Must travel 90% of arm length
        self.THRESH_LEAN_BACK = 20.0 # Max back lean angle allowed (Degrees)
        self.THRESH_SHALLOW_DEPTH = 0.1 # Max distance from shoulder to bar at bottom
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_bar_y = 0
        self.velocity = 0
        self.prev_velocity = 0 # For acceleration
        self.max_bar_y = 0  # Track highest point in rep
        self.start_x = 0       # Track horizontal path
        
        # Timers
        self.start_time = 0

    def process(self, landmarks, raw_landmarks=None):
        """
        Main Logic Pipeline.
        Args:
            landmarks: List of Landmark objects (Corrected by Normalizer).
            raw_landmarks: Raw MediaPipe landmarks for visualization.
        """
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        shoulder = self._get_midpoint(landmarks[11], landmarks[12])
        elbow = self._get_midpoint(landmarks[13], landmarks[14])
        wrist = self._get_midpoint(landmarks[15], landmarks[16])
        hip = self._get_midpoint(landmarks[23], landmarks[24])
        
        # --- STEP 2: CALCULATE METRICS ---
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        torso_angle = self._calculate_angle(hip, shoulder, elbow)

        curr_bar_y = wrist.y
        dt = 1.0 / self.FPS
        
        if self.prev_bar_y != 0:
            self.velocity = (curr_bar_y - self.prev_bar_y) / dt
        self.prev_bar_y = curr_bar_y
        
        acceleration = (self.velocity - self.prev_velocity) / dt if self.prev_velocity != 0 else 0
        self.prev_velocity = self.velocity

        wrist_diff = abs(landmarks[15].y - landmarks[16].y)
        if wrist_diff > 0.05:
             self._add_fault("ASYMMETRY", 10, "Push Evenly!")

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            
            # TRIGGER: Bar moves up
            norm_vel = self.velocity / self.arm_length
            if norm_vel < -0.2:
                self._start_rep(wrist.x)
                self.state = "ASCENDING"

        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push!"
            
            if curr_bar_y < self.max_bar_y: 
                 self.max_bar_y = curr_bar_y

            # FAULT: Lean Back
            if torso_angle < (90 - self.THRESH_LEAN_BACK):
                self._add_fault("LEAN_BACK", 10, "Don't lean back!")

            # TRANSITION: Velocity flips to positive (Moving Down)
            norm_vel = self.velocity / self.arm_length
            if norm_vel > 0.1:
                self.state = "DESCENDING"
                
                # CHECK LOCKOUT
                if elbow_angle < 160:
                    self._add_fault("INCOMPLETE_LOCKOUT", 5, "Fully lock out!")

        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control down..."
            
            # TRANSITION: Bar returns to shoulder height
            if curr_bar_y >= self.shoulder_y:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            # CHECK DEPTH
            depth_gap = abs(curr_bar_y - self.shoulder_y)
            if depth_gap > self.THRESH_SHALLOW_DEPTH * self.arm_length:
                self._add_fault("SHALLOW_DEPTH", 5, "Bring the bar to your shoulders!")

            self._finish_rep()
            self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
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

    # --- HELPERS ---

    def _start_rep(self, current_x):
        """Reset score and faults for new rep."""
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()
        self.max_bar_y = self.shoulder_y
        self.start_x = current_x # Record where the bar started (Shoulder line)

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per rep."""
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        """Finalizes the rep."""
        # Only count if Form > 50 (Not a complete failure)
        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"

    def _get_midpoint(self, p1, p2):
        """Returns a dummy landmark representing the center of two points."""
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
        return Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)

    def _calculate_angle(self, a, b, c):
        """Standard 2D angle math."""
        # a=First, b=Vertex, c=End
        # For Elbow Angle: Shoulder(a) -> Elbow(b) -> Wrist(c)
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip to prevent numerical errors (arccos requires -1 to 1)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)