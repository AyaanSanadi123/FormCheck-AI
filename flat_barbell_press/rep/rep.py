import numpy as np
import time

class BenchPressRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.bench_y = calibration_data.get('bench_y', 0)     # Shoulder height (Lying down)
        self.arm_length = calibration_data.get('arm_length', 1.0) # Scale factor
        self.facing_side = calibration_data.get('facing_side', 1) # 1=Right, -1=Left
        
        # Thresholds (Normalized to Arm Length)
        self.THRESH_ROM = 0.90       # Must travel 90% of arm length
        self.THRESH_BRIDGE = 0.15    # Max hip lift allowed (15% of arm length)
        self.THRESH_BOUNCE = 3.5     # Acceleration spike threshold (m/s^2)
        self.THRESH_FLARE = 85.0     # Max Elbow Angle allowed (Degrees)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Unrack Bar"
        
        # Physics Tracking
        self.prev_bar_y = 0
        self.velocity = 0
        self.prev_velocity = 0 # For acceleration (bounce detection)
        self.min_bar_y = 1000  # Track lowest point in rep
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
        # 11=L_Sh, 12=R_Sh, 13=L_Elb, 14=R_Elb, 15=L_Wrist, 16=R_Wrist, 23=L_Hip, 24=R_Hip
        
        # Average Left/Right for "Main Logic" (Simulates Bar Center)
        shoulder = self._get_midpoint(landmarks[11], landmarks[12])
        elbow = self._get_midpoint(landmarks[13], landmarks[14])
        wrist = self._get_midpoint(landmarks[15], landmarks[16])
        hip = self._get_midpoint(landmarks[23], landmarks[24])
        
        # --- STEP 2: CALCULATE METRICS ---
        
        # A. Elbow Angle (Lockout Check)
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        
        # B. Bar Velocity (Vertical)
        curr_bar_y = wrist.y # Y is Gravity
        dt = 1.0 / self.FPS
        
        if self.prev_bar_y != 0:
            self.velocity = (curr_bar_y - self.prev_bar_y) / dt
        self.prev_bar_y = curr_bar_y
        
        # C. Acceleration (For Bounce Detection)
        acceleration = (self.velocity - self.prev_velocity) / dt if self.prev_velocity != 0 else 0
        self.prev_velocity = self.velocity

        # D. Asymmetry Check (Always Active)
        # Check if one arm is lagging behind the other (Tilt)
        wrist_diff = abs(landmarks[15].y - landmarks[16].y)
        if wrist_diff > 0.05: # Threshold for tilt
             self._add_fault("ASYMMETRY", 10, "Push Evenly!")

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        # A. IDLE / TOP (Waiting for descent)
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            
            # TRIGGER: 
            # 1. Elbows unlock (< 165)
            # 2. Bar moves down (Vel > 0.2 normalized)
            # 3. Bar is ALIGNED with shoulders (Wrist X approx Shoulder X)
            #    This ensures they don't start the rep until the bar is "set" over the pivot.
            norm_vel = self.velocity / self.arm_length
            alignment_error = abs(wrist.x - shoulder.x)
            
            if elbow_angle < 165 and norm_vel > 0.2:
                if alignment_error < (0.10 * self.arm_length):
                    self._start_rep(wrist.x)
                    self.state = "DESCENDING"
                else:
                    self.feedback_buffer = "Set Bar Over Shoulders"

        # B. DESCENDING (Eccentric)
        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control Down..."
            
            # TRACK: Lowest Point (Max Y value)
            if curr_bar_y > self.min_bar_y: 
                 self.min_bar_y = curr_bar_y
            
            # FAULT: Glute Bridge (Cheating)
            if (self.bench_y - hip.y) > self.THRESH_BRIDGE: 
                 self._add_fault("GLUTE_BRIDGE", 10, "Keep Hips Down!")

            # FAULT: Elbow Flare (Safety)
            flare_angle = self._calculate_angle(hip, shoulder, elbow)
            if flare_angle > self.THRESH_FLARE:
                self._add_fault("ELBOW_FLARE", 5, "Tuck Elbows!")

            # TRANSITION: Velocity flips to negative (Moving Up)
            norm_vel = self.velocity / self.arm_length
            if norm_vel < -0.1:
                self.state = "ASCENDING"
                
                # CHECK DEPTH (SHALLOW)
                # Distance from Chest (Bench Y) to Lowest Point (Min Bar Y)
                # Ideally Min Bar Y ~= Bench Y. If Min Bar Y is smaller (higher up), gap exists.
                depth_gap = self.bench_y - self.min_bar_y
                # If gap is positive and large, they didn't touch chest
                if depth_gap > (0.15 * self.arm_length):
                    self._add_fault("SHALLOW", 5, "Touch Your Chest!")

                # Check for Bounce (High Acceleration spike at turnaround)
                if abs(acceleration) > self.THRESH_BOUNCE:
                    self._add_fault("BOUNCE", 5, "Don't Bounce!")

        # C. ASCENDING (Concentric)
        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push Back!"
            
            # FAULT: Bar Path (Vertical / Guillotine Check)
            # In a good bench press, the bar moves horizontally towards the shoulders (Head).
            # We check the horizontal distance between Start X (Top) and Current X.
            # Ideally, the bar travels in a curve. If X never changes, it's a straight line (Bad).
            # Note: This is a simplified check.
            drift = abs(wrist.x - self.start_x)
            if drift < 0.02: # Very strict vertical line
                 self._add_fault("BAD_PATH", 5, "Push Back Towards Face")
            
            # TRANSITION: Elbows Locked Out
            norm_vel = self.velocity / self.arm_length
            if elbow_angle > 165 and abs(norm_vel) < 0.1:
                self.state = "COMPLETE"

        # D. COMPLETE (Rep Done)
        elif self.state == "COMPLETE":
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
        self.min_bar_y = 0
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