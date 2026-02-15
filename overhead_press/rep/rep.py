import numpy as np
import time

class OverheadPressRepCounter:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.shoulder_y = calibration_data.get('shoulder_y', 0)     # Shoulder height (Standing)
        self.arm_length = calibration_data.get('arm_length_scale', 1.0) # Scale factor from gatekeeper
        
        # Thresholds (Normalized to Arm Length)
        self.THRESH_ROM = 0.90       # Must travel 90% of arm length
        self.THRESH_LEAN_BACK = 20.0 # Max back lean angle allowed (Degrees)
        self.THRESH_SHALLOW_DEPTH = 0.1 # Max distance from shoulder to bar at bottom
        self.THRESH_DESCENT_SPEED = 0.8 # Normalized velocity threshold for controlled descent
        self.THRESH_DIRECTION_CHANGE = 0.1 # Normalized velocity for early direction change detection (e.g., bar stops before lockout)
        
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
        self.prev_time = 0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Normalized) ---
        shoulder = self._get_midpoint(landmarks[11], landmarks[12])
        elbow = self._get_midpoint(landmarks[13], landmarks[14])
        wrist = self._get_midpoint(landmarks[15], landmarks[16])
        hip = self._get_midpoint(landmarks[23], landmarks[24])

        dt = 0
        if timestamp and self.prev_time:
            dt = timestamp - self.prev_time
        self.prev_time = timestamp
        if dt <= 0: dt = 0.001 # Avoid division by zero


        # --- STEP 2: CALCULATE METRICS ---
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        torso_angle = self._calculate_angle(hip, shoulder, elbow)

        curr_bar_y = wrist.y
        
        if self.prev_bar_y != 0:
            self.velocity = (curr_bar_y - self.prev_bar_y) / dt
        self.prev_bar_y = curr_bar_y
        
        acceleration = (self.velocity - self.prev_velocity) / dt if self.prev_velocity != 0 else 0
        self.prev_velocity = self.velocity

        # Asymmetry check (needs both wrists if raw_landmarks is available)
        # Note: In normalized landmarks, 15 and 16 are Left and Right wrist.
        wrist_diff = abs(landmarks[15].y - landmarks[16].y)
        if wrist_diff > 0.05:
             self._add_fault("ASYMMETRY", 10, "Push Evenly!")

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            
            # TRIGGER: Bar moves up
            norm_vel = self.velocity / self.arm_length
            if norm_vel < -0.2:
                self._reset_rep(curr_bar_y) # Pass curr_bar_y for initial max_bar_y
                self.state = "ASCENDING"

        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push!"
            
            # Track highest point
            if curr_bar_y < self.max_bar_y: 
                 self.max_bar_y = curr_bar_y

            # FAULT: Lean Back
            if torso_angle < (90 - self.THRESH_LEAN_BACK):
                self._add_fault("LEAN_BACK", 10, "Don't lean back!")

            # FAULT: Early Descent / Bar stops before lockout (PARTIAL_ROM)
            # Check if bar velocity becomes positive (starts moving down) too early
            norm_vel = self.velocity / self.arm_length
            if norm_vel > self.THRESH_DIRECTION_CHANGE and elbow_angle < 170: # 170deg as near lockout
                self._add_fault("PARTIAL_ROM", 15, "Fully Extend at Top!")

            # TRANSITION: Velocity flips to positive (Moving Down)
            if norm_vel > 0.1:
                self.state = "DESCENDING"
                
                # CHECK LOCKOUT (once ascending phase is completed)
                if elbow_angle < 160:
                    self._add_fault("INCOMPLETE_LOCKOUT", 5, "Fully lock out!")

        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control down..."
            
            # FAULT: Uncontrolled Descent Speed
            norm_vel = self.velocity / self.arm_length
            if norm_vel > self.THRESH_DESCENT_SPEED:
                self._add_fault("CONTROL", 10, "Lower Slowly!")

            # TRANSITION: Bar returns to shoulder height
            if curr_bar_y >= self.shoulder_y:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            # CHECK DEPTH
            depth_gap = abs(curr_bar_y - self.shoulder_y)
            if depth_gap > self.THRESH_SHALLOW_DEPTH * self.arm_length:
                self._add_fault("SHALLOW_DEPTH", 5, "Bring the bar to your shoulders!")

            self._finalize_rep_success()
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

    def _reset_rep(self, initial_bar_y):
        """Reset score and faults for new rep."""
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_bar_y = initial_bar_y # Initialize with current bar position

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per rep."""
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finalize_rep_success(self):
        """Finalizes the rep with nuanced feedback."""
        # Rep only counts if score is salvageable and some effort was made
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback_buffer = "Rep Failed - Form too poor"

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