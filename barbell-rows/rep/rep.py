import numpy as np
from collections import deque

class BarbellRowRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.setup_torso_angle = calibration_data.get('setup_torso_angle', 45.0)
        self.torso_length = calibration_data.get('torso_length', 1.0)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Hinge and hold."
        
        # Physics Tracking
        self.prev_wrist_y = 0.0
        self.velocity = 0.0
        self.velocity_history = deque(maxlen=5) # Smooth over 5 frames
        self.start_wrist_y = 0.0
        self.max_wrist_y = 0.0
        self.min_wrist_distance_to_torso = 999.0
        
        # Drift Compensation
        self.lowest_point_y = 0.0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT JOINTS (Normalized) ---
        # Normalizer centers X on Ankles, Y=0 is Floor, +Y is UP.
        l_sh = landmarks[11]; r_sh = landmarks[12]
        l_hip = landmarks[23]; r_hip = landmarks[24]
        l_wr = landmarks[15]; r_wr = landmarks[16]
        
        sh_x = (l_sh.x + r_sh.x) / 2
        sh_y = (l_sh.y + r_sh.y) / 2
        hip_x = (l_hip.x + r_hip.x) / 2
        hip_y = (l_hip.y + r_hip.y) / 2
        wr_x = (l_wr.x + r_wr.x) / 2
        wr_y = (l_wr.y + r_wr.y) / 2

        # --- STEP 2: CALCULATE METRICS ---
        # A. Smoothed Velocity (UP is Positive)
        dt = 1.0 / self.FPS
        raw_vel = (wr_y - self.prev_wrist_y) / dt
        self.velocity_history.append(raw_vel)
        self.velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        self.prev_wrist_y = wr_y

        # B. Current Torso Angle
        dx = abs(hip_x - sh_x)
        dy = abs(hip_y - sh_y)
        current_torso_angle = np.degrees(np.arctan2(dy, dx))

        # C. Distance from Wrist to Torso Line (Shoulder -> Hip)
        # Using standard point-to-line distance formula
        wrist_to_torso_dist = self._distance_to_line(
            (wr_x, wr_y), (sh_x, sh_y), (hip_x, hip_y)
        )

        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (Dead Hang)
        if self.state == "IDLE":
            self.feedback_buffer = "Pull to your stomach"
            
            # Update lowest point while idle to handle drift
            if wr_y < self.lowest_point_y:
                self.lowest_point_y = wr_y

            # TRIGGER: Velocity positive and moves up a bit from lowest point
            if self.velocity > 0.15 and wr_y > (self.lowest_point_y + 0.05):
                self._start_rep(wr_y)
                self.state = "PULLING"

        # B. PULLING (Concentric - Bar moving UP)
        elif self.state == "PULLING":
            self.feedback_buffer = "Squeeze shoulder blades!"
            
            # Track max height and min distance to torso
            if wr_y > self.max_wrist_y:
                self.max_wrist_y = wr_y
            if wrist_to_torso_dist < self.min_wrist_distance_to_torso:
                self.min_wrist_distance_to_torso = wrist_to_torso_dist

            # FAULT: Torso Heave (Standing up to cheat)
            if current_torso_angle > (self.setup_torso_angle + 15.0):
                self._add_fault("TORSO_HEAVE", 10, "Keep back flat! Don't heave.")

            # TRANSITION: Velocity flips negative (starts going down)
            if self.velocity < -0.1:
                # CHECK SHORT PULL HERE
                # If the bar didn't get within ~15% of torso length to the body
                if self.min_wrist_distance_to_torso > (self.torso_length * 0.15):
                    self._add_fault("SHORT_PULL", 5, "Pull the bar all the way to your body!")
                
                self.state = "RETURNING"

        # C. RETURNING (Eccentric - Bar moving DOWN)
        elif self.state == "RETURNING":
            self.feedback_buffer = "Control the weight down"
            
            # TRANSITION: Bar returns to near starting height
            # Use a dynamic baseline: slightly above the start is okay
            if wr_y <= (self.start_wrist_y + 0.08):
                self._finish_rep(success=True)
                self.state = "IDLE"
                
            # EARLY REVERSAL (No Extension Fault)
            # If velocity flips positive again before reaching the bottom
            elif self.velocity > 0.15:
                self._add_fault("NO_EXTENSION", 5, "Let arms fully hang between reps!")
                self._finish_rep(success=False) # Count as failed/bad rep immediately
                self.state = "IDLE"

        # D. COMPLETE (Transient state handled by _finish_rep -> IDLE)
        
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
            "torso_angle": int(current_torso_angle)
        }

    # --- HELPERS ---
    def _start_rep(self, start_y):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_wrist_y = start_y
        self.max_wrist_y = start_y
        self.min_wrist_distance_to_torso = 999.0

    def _add_fault(self, code, penalty, msg):
        # Don't add duplicate faults for the same rep
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self, success=True):
        if success and self.current_score > 50:
            self.rep_count += 1
            # Reset drift baseline
            self.lowest_point_y = self.start_wrist_y
        elif not success:
            self.feedback_buffer = "Rep Failed (Bad Form)"
            # Force reset lowest point to current to avoid immediate re-trigger
            self.lowest_point_y = self.prev_wrist_y 

    def _distance_to_line(self, p0, p1, p2):
        """Calculates the shortest distance from point p0 to the line defined by p1 and p2."""
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        
        # Numerator: |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)|
        num = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
        # Denominator: sqrt((x2 - x1)^2 + (y2 - y1)^2)
        den = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if den == 0:
            return 0
        return num / den