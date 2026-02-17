import numpy as np
from collections import deque

class SeatedRowRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.setup_torso_angle = calibration_data.get('setup_torso_angle', 90.0)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Reach and brace."
        
        # Physics Tracking (Horizontal X-Axis)
        self.prev_wrist_x = None # Initialized on first frame
        self.velocity_x = 0.0
        self.velocity_history = deque(maxlen=5) # Smooth over 5 frames
        
        # Drift Compensation & Rep Bounds
        self.start_wrist_x = 0.0
        self.min_wrist_x = 999.0 # Tracks how close the bar gets to the body
        self.furthest_reach_x = 0.0 # Tracks the stretch
        self.start_shoulder_y = 0.0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT JOINTS (Active Side) ---
        if self.active_side == "LEFT":
            idx_sh, idx_wr = 11, 15
        else:
            idx_sh, idx_wr = 12, 16

        sh = landmarks[idx_sh]
        wr = landmarks[idx_wr]
        
        sh_x, sh_y = sh.x, sh.y
        wr_x, wr_y = wr.x, wr.y

        # Handle first frame to avoid velocity spike
        if self.prev_wrist_x is None:
            self.prev_wrist_x = wr_x
            return None # Skip calculation for first frame

        # --- STEP 2: CALCULATE METRICS ---
        # A. Smoothed Velocity X
        # PULLING = Negative Velocity (Moving from +X towards 0)
        # RETURNING = Positive Velocity (Moving from 0 towards +X)
        dt = 1.0 / self.FPS
        raw_vel_x = (wr_x - self.prev_wrist_x) / dt
        self.velocity_history.append(raw_vel_x)
        self.velocity_x = np.mean(self.velocity_history) if self.velocity_history else 0.0
        self.prev_wrist_x = wr_x

        # B. Current Torso Angle (Hip is at 0,0 so dx=sh_x, dy=sh_y)
        # Calculate angle from horizontal
        dx = sh_x
        dy = sh_y
        if dx == 0: dx = 0.001
        current_torso_angle = np.degrees(np.arctan2(dy, dx))
        if current_torso_angle < 0: current_torso_angle += 360

        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (Full Stretch)
        if self.state == "IDLE":
            self.feedback_buffer = "Pull to your stomach"
            
            # Update furthest reach to handle drift
            if wr_x > self.furthest_reach_x:
                self.furthest_reach_x = wr_x

            # TRIGGER: Velocity is distinctly negative (pulling in)
            if self.velocity_x < -0.15 and wr_x < (self.furthest_reach_x - 0.05):
                # Use furthest_reach_x as the baseline for a full rep
                self._start_rep(self.furthest_reach_x, sh_y)
                self.state = "PULLING"

        # B. PULLING (Concentric - Bar moving IN)
        elif self.state == "PULLING":
            self.feedback_buffer = "Drive elbows back!"
            
            # Track closest approach to torso
            if wr_x < self.min_wrist_x:
                self.min_wrist_x = wr_x

            # FAULT: Momentum Swing (Leaning back too far)
            if current_torso_angle > (self.setup_torso_angle + 15.0) or current_torso_angle > 110.0:
                self._add_fault("MOMENTUM_SWING", 10, "Don't lean back! Keep torso still.")
                
            # FAULT: Shrugging (Shoulders moving up relative to hip/start)
            # Normalized Y increases UPWARDS. So Shrugging = Higher Y.
            if sh_y > (self.start_shoulder_y + 0.1): # 0.1 torso units up
                self._add_fault("SHRUGGING", 5, "Keep shoulders down!")

            # TRANSITION: Velocity flips positive (starts going away from body)
            if self.velocity_x > 0.1:
                # CHECK SHORT PULL HERE
                # Wrist should get very close to X=0 (Hip). If it stops > 0.2 units away:
                if self.min_wrist_x > 0.2:
                    self._add_fault("SHORT_PULL", 5, "Pull the handle all the way to your torso!")
                
                self.state = "RETURNING"

        # C. RETURNING (Eccentric - Bar moving OUT)
        elif self.state == "RETURNING":
            self.feedback_buffer = "Control the weight back"
            
            # TRANSITION: Bar returns to near starting stretch
            if wr_x >= (self.start_wrist_x - 0.08):
                self._finish_rep(success=True)
                self.state = "IDLE"
                
            # EARLY REVERSAL (Short eccentric / Bouncing the weight)
            elif self.velocity_x < -0.15:
                self._add_fault("NO_STRETCH", 5, "Let arms fully extend for the stretch!")
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
            "velocity": self.velocity_x,
            "faults": list(set([f['code'] for f in self.faults])),
            "torso_angle": int(current_torso_angle)
        }

    # --- HELPERS ---
    def _start_rep(self, start_x, start_sh_y):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_wrist_x = start_x
        self.min_wrist_x = start_x
        self.start_shoulder_y = start_sh_y

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self, success=True):
        if success and self.current_score > 50:
            self.rep_count += 1
            self.furthest_reach_x = self.start_wrist_x
        elif not success:
            self.feedback_buffer = "Rep Failed (Bad Form)"
            self.furthest_reach_x = self.prev_wrist_x