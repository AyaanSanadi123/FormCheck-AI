import numpy as np
from collections import deque

class TricepPushdownRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.facing_side = calibration_data.get('facing_side', 1.0)
        self.shoulder_origin_x = calibration_data.get('shoulder_origin_x', 0.5)
        self.shoulder_origin_y = calibration_data.get('shoulder_origin_y', 0.5)
        self.arm_length = calibration_data.get('arm_length', 1.0)
        
        # Calculate Normalized Baseline (The user's specific starting elbow position)
        raw_base_el_x = calibration_data.get('baseline_el_x', 0.5)
        raw_base_el_y = calibration_data.get('baseline_el_y', 0.5) # Gatekeeper sends raw Y
        
        # Avoid division by zero
        if self.arm_length < 0.001: self.arm_length = 1.0
        
        # Normalize the baseline coordinates to match the incoming frame data
        # Formula must match Normalizer: (x - origin) * facing / scale
        self.baseline_el_x = ((raw_base_el_x - self.shoulder_origin_x) * self.facing_side) / self.arm_length
        # Formula must match Normalizer: (origin - y) / scale (Inverted Y)
        self.baseline_el_y = (self.shoulder_origin_y - raw_base_el_y) / self.arm_length
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Push down and lock out."
        
        # Physics Tracking
        self.prev_angle = None 
        self.angular_velocity = 0.0
        self.velocity_history = deque(maxlen=5) 
        
        # Rep Bounds
        self.max_extension_angle = 0.0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT ACTIVE JOINTS ---
        # Shoulder is (0,0). User faces RIGHT (+X). UP is +Y.
        # The Normalizer returns a full list, so indices correspond to MediaPipe topology.
        if self.active_side == "LEFT":
            sh = landmarks[11]; el = landmarks[13]; wr = landmarks[15]
        else:
            sh = landmarks[12]; el = landmarks[14]; wr = landmarks[16]

        # --- STEP 2: CALCULATE METRICS ---
        # A. Elbow Angle (Interior angle of Sh -> El -> Wr)
        current_angle = self._calculate_angle(sh, el, wr)
        
        if self.prev_angle is None:
            self.prev_angle = current_angle
            return None # Skip first frame velocity spike

        # B. Smoothed Angular Velocity (Degrees per second)
        # Positive = Extending (Pushing down). Negative = Flexing (Returning).
        dt = 1.0 / self.FPS
        raw_vel = (current_angle - self.prev_angle) / dt
        self.velocity_history.append(raw_vel)
        self.angular_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        self.prev_angle = current_angle

        # C. Elbow Drift (Euclidean distance from baseline in "Arm Units")
        # Both points are now in the same Normalized Coordinate System
        elbow_drift = np.sqrt((el.x - self.baseline_el_x)**2 + (el.y - self.baseline_el_y)**2)

        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (Elbows bent, hands high)
        if self.state == "IDLE":
            self.feedback_buffer = "Push down and lock out"
            
            # TRIGGER: Extending arm rapidly
            # 100 degrees is slightly open, meaning movement has started
            if self.angular_velocity > 15.0 and current_angle > 100.0:
                self._start_rep(current_angle)
                self.state = "PUSHING"

        # B. PUSHING (Concentric - Hands moving DOWN)
        elif self.state == "PUSHING":
            self.feedback_buffer = "Squeeze the tricep!"
            
            if current_angle > self.max_extension_angle:
                self.max_extension_angle = current_angle

            # FAULT: Elbow Swing (Breaking the anchor)
            # 0.20 means the elbow moved 20% of an upper arm length away from baseline
            if elbow_drift > 0.20:
                self._add_fault("ELBOW_SWING", 10, "Keep elbows pinned to your sides!")

            # TRANSITION: Reversing direction (Velocity flips negative)
            # -5.0 allows for a brief pause at the bottom without triggering immediately
            if self.angular_velocity < -5.0 or (current_angle < self.max_extension_angle - 10.0):
                # CHECK SHORT LOCKOUT
                # Perfect lockout is ~180. We require at least 160.
                if self.max_extension_angle < 160.0:
                    self._add_fault("SHORT_LOCKOUT", 5, "Lock your arms out completely!")
                
                self.state = "RETURNING"

        # C. RETURNING (Eccentric - Hands moving UP)
        elif self.state == "RETURNING":
            self.feedback_buffer = "Control it up"
            
            # FAULT: Elbow Swing (often happens during eccentric as user lets weight pull them)
            if elbow_drift > 0.20:
                self._add_fault("ELBOW_SWING", 10, "Don't let the weight pull your elbows up!")
            
            # TRANSITION: Arm is fully bent again
            if current_angle <= 100.0:
                self._finish_rep(success=True)
                self.state = "IDLE"
                
            # EARLY REVERSAL (Half-repping at the bottom)
            elif self.angular_velocity > 15.0:
                self._add_fault("HALF_REP", 5, "Let your hands come all the way up!")
                self._finish_rep(success=False) 
                self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "coords": landmarks,          
            "raw_coords": raw_landmarks,  
            "velocity": self.angular_velocity,
            "faults": list(set([f['code'] for f in self.faults])),
            "elbow_angle": int(current_angle),
            "elbow_drift": elbow_drift,
            "active_side": self.active_side
        }

    # --- HELPERS ---
    def _start_rep(self, start_ang):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_extension_angle = start_ang

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self, success=True):
        if success and self.current_score > 50:
            self.rep_count += 1
        elif not success:
            self.feedback_buffer = "Rep Failed (Bad Form)"

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0: return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))
