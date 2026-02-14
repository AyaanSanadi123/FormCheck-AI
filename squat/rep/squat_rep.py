import numpy as np
import time

class SquatRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.floor_y = calibration_data.get('floor_y', 0.5)
        self.calibrated_scale = calibration_data.get('calibrated_scale', 1.0)
        self.calibrated_angle = calibration_data.get('calibrated_angle', 90.0)

        # Thresholds
        self.THRESH_DEPTH = 100    # Degrees (Standard Parallel)
        self.THRESH_STAND = 165    # Degrees (Lockout)
        self.THRESH_VALGUS = 0.05  # Normalized distance (Knee vs Ankle Z-alignment)
        self.THRESH_HEEL = 0.04    # Normalized Y-distance (Heel Lift)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []           # List of faults for current rep
        self.feedback_buffer = "Ready" 
        
        # Smoothing & Velocity
        self.prev_hip_y = 0
        self.prev_shoulder_y = 0
        self.velocity = 0 # Hip vertical velocity (scalar for output)
        self.hip_velocity_signed = 0 # Signed (+ is down, - is up)
        self.shoulder_velocity_signed = 0
        self.start_time = 0
        self.bottom_time = 0

    def process(self, landmarks, raw_landmarks=None):
        """
        Main Pipeline: Input -> Analyze -> Output
        Args:
            landmarks: Normalized/Aligned landmarks (Objects with .x, .y, .z)
            raw_landmarks: Optional Raw MediaPipe landmarks for environmental checks.
        """
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS (Aligned) ---
        if self.active_side == "LEFT":
            sh = landmarks[11]
            hip = landmarks[23]
            knee = landmarks[25]
            ankle = landmarks[27]
        else:
            sh = landmarks[12]
            hip = landmarks[24]
            knee = landmarks[26]
            ankle = landmarks[28]

        # --- STEP 2: STABILITY WATCHDOG (Raw Data) ---
        # Check if the user is twisting (Major Form Failure)
        # For simplicity, we assume Gatekeeper stability holds,
        # but we can add a check for active side vs other side hip visibility ratio later.
        is_unstable = False

        # --- STEP 3: METRICS CALCULATION ---
        # Calculate Angles
        knee_angle = self._calculate_angle(hip, knee, ankle)
        hip_angle = self._calculate_angle(sh, hip, knee) # New: Explicit Hip Angle
        
        # Calculate Velocities (Signed: +Down, -Up)
        curr_time = time.time()
        dt = 1.0 / self.FPS # Approx delta time
        
        # Hip Velocity
        if self.prev_hip_y != 0:
            self.hip_velocity_signed = (hip.y - self.prev_hip_y) / dt
            self.velocity = abs(self.hip_velocity_signed)
        self.prev_hip_y = hip.y
        
        # Shoulder Velocity
        if self.prev_shoulder_y != 0:
            self.shoulder_velocity_signed = (sh.y - self.prev_shoulder_y) / dt
        self.prev_shoulder_y = sh.y

        # --- STEP 4: STATE MACHINE & FAULT DETECTION ---
        
        # A. IDLE / TOP (Waiting for rep)
        if self.state == "IDLE":
            self.feedback_buffer = "Stand Tall" if not is_unstable else "Fix Hips"
            
            # TRIGGER: Hip Angle breaks < 165 AND Body is moving down
            # Using multiple checks to prevent false alarms
            if (hip_angle < 165 and 
                self.hip_velocity_signed > 0.1 and 
                self.shoulder_velocity_signed > 0.1):
                
                self._start_rep()
                self.state = "DESCENDING"

        # B. DESCENDING (Going Down)
        elif self.state == "DESCENDING":
            self.feedback_buffer = "Control Down..."
            
            # CHECK: Knee Valgus (Safety) [-10 pts]
            self._check_valgus(knee, ankle)
            
            # FAULT: Dive Bombing (Too Fast) [-5 pts]
            if self.velocity > 0.85: 
                self._add_fault("TOO_FAST", 5, "Slow Down!")
                
            # FAULT: Heel Lift (Check against Floor Baseline) [-10 pts]
            if raw_landmarks:
                if self.active_side == "LEFT":
                    idx_heel = 29
                else:
                    idx_heel = 30
                
                # Safe access for Raw Landmarks
                raw_heel = raw_landmarks[idx_heel]
                raw_heel_y = getattr(raw_heel, 'y', raw_heel.get('y') if isinstance(raw_heel, dict) else 0)

                if (self.floor_y - raw_heel_y) > self.THRESH_HEEL:
                    self._add_fault("HEEL_LIFT", 10, "Keep Heels Flat!")

            # Transition to BOTTOM
            if knee_angle < self.THRESH_DEPTH:
                self.state = "BOTTOM"
                self.bottom_time = curr_time
                
            # Abort (User stood back up without squatting)
            elif hip_angle > self.THRESH_STAND:
                self.state = "IDLE" 

        # C. BOTTOM (The Hole)
        elif self.state == "BOTTOM":
            self.feedback_buffer = "Drive Up!"
            
            # FAULT: Depth Check (Strict) [-5 pts]
            if hip.y < knee.y: 
                self._add_fault("SHALLOW", 5, "Hit Parallel!")

            # FAULT: Butt Wink (Lumbar Flexion) [-10 pts]
            torso_angle = self._calculate_angle(sh, hip, knee)
            if torso_angle < 70: 
                self._add_fault("ROUNDING", 10, "Chest Up!")

            # Transition to ASCENT
            # Trigger: Hip angle increases from bottom (proxy via knee angle > depth + buffer)
            if knee_angle > self.THRESH_DEPTH + 10:
                self.state = "ASCENDING"

        # D. ASCENDING (Coming Up)
        elif self.state == "ASCENDING":
            self.feedback_buffer = "Push!"
            
            # CHECK: Knee Valgus (Safety) [-10 pts]
            self._check_valgus(knee, ankle)

            # FAULT: Good Morning Squat [-5 pts]
            # Check if Hips rising significantly faster than Shoulders
            # Both velocities should be negative (UP). 
            # If hip_vel is -1.0 (fast up) and shoulder_vel is -0.2 (slow up)
            if (self.hip_velocity_signed < -0.2 and 
                self.shoulder_velocity_signed < 0 and 
                abs(self.hip_velocity_signed) > abs(self.shoulder_velocity_signed) * 1.5):
                
                self._add_fault("GOOD_MORNING", 5, "Chest Up First!")
            
            # Transition to COMPLETE
            if hip_angle > self.THRESH_STAND and abs(self.velocity) < 0.2:
                self.state = "COMPLETE"

        # E. COMPLETE (Rep Done)
        elif self.state == "COMPLETE":
            self._finish_rep()
            self.state = "IDLE"

        # --- STEP 5: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "angle": int(knee_angle),
            "coords": landmarks,      
            "raw_coords": raw_landmarks,          
            "velocity": self.velocity,
            "faults": list(set([f['code'] for f in self.faults])) 
        }

    # --- HELPERS ---

    def _start_rep(self):
        """Reset score and faults for new rep."""
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_time = time.time()

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per phase."""
        if any(f['code'] == code for f in self.faults):
            return
            
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        """Finalizes the rep."""
        # Safety Threshold: Score < 70 = BAD (Do not count or flag as fail)
        if self.current_score >= 70:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"

    def _check_valgus(self, knee, ankle):
        """
        Checks for Knee Valgus (Knees caving in).
        Uses Z-axis deviation in Aligned View. 
        In a correct squat, Knee Z should track closely with Ankle Z.
        """
        z_deviation = abs(knee.z - ankle.z)
        if z_deviation > self.THRESH_VALGUS:
            self._add_fault("KNEE_VALGUS", 10, "Knees Out!")

    def _calculate_angle(self, a, b, c):
        """Standard 2D angle math using Aligned coordinates."""
        # Using .x and .y directly
        a_arr = np.array([a.x, a.y])
        b_arr = np.array([b.x, b.y]) # Vertex
        c_arr = np.array([c.x, c.y])
        
        ba = a_arr - b_arr
        bc = c_arr - b_arr
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
