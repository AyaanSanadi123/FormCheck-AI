import numpy as np

class LatPullRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # We don't need max_reach_y here because Normalizer already set it to 0.0
        # Normalizer also scaled everything by Shoulder Width.
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Prepare to Pull"
        
        # Physics Tracking
        self.prev_wrist_y = 0.0
        self.velocity = 0.0
        self.max_wrist_y = 0.0  # Tracks depth of current rep
        
        # Dynamic Baselines (Captured at Start of Rep)
        self.target_depth_y = 0.0
        self.start_torso_length = 0.0
        self.ignore_rep = False

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS ---
        # Note: Normalizer aligned spine to X=0 and top reach to Y=0.
        l_sh = landmarks[11]; r_sh = landmarks[12]
        l_wr = landmarks[15]; r_wr = landmarks[16]
        l_hip = landmarks[23]; r_hip = landmarks[24]
        
        avg_sh_y = (l_sh.y + r_sh.y) / 2
        avg_wr_y = (l_wr.y + r_wr.y) / 2
        avg_hip_y = (l_hip.y + r_hip.y) / 2
        
        # Current Torso Length (Vertical 2D distance)
        current_torso_len = avg_hip_y - avg_sh_y
        
        # --- STEP 2: CALCULATE METRICS ---
        dt = 1.0 / self.FPS
        # Velocity > 0 means pulling DOWN (because Y increases downwards)
        self.velocity = (avg_wr_y - self.prev_wrist_y) / dt
        self.prev_wrist_y = avg_wr_y

        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (At the Top)
        if self.state == "IDLE":
            self.feedback_buffer = "Pull down and squeeze"
            
            # TRIGGER: Wrists move down past 10% of shoulder depth
            # (avg_sh_y represents roughly 100% ROM target)
            if avg_wr_y > (avg_sh_y * 0.1):
                self._start_rep(current_torso_len, avg_sh_y)
                self.state = "PULLING"

        # B. PULLING (Concentric - Bar moving down)
        elif self.state == "PULLING":
            self.feedback_buffer = "Drive elbows down!"
            
            # Track max depth
            if avg_wr_y > self.max_wrist_y:
                self.max_wrist_y = avg_wr_y

            # FAULT: Asymmetric Pull
            if abs(l_wr.y - r_wr.y) > 0.15: # 15% of shoulder width difference
                self._add_fault("ASYMMETRIC_PULL", 10, "Keep the bar level!")

            # FAULT: Momentum Swing (Leaning Back)
            # If torso length shrinks significantly, user is leaning away from camera
            if current_torso_len < (self.start_torso_length * 0.85):
                self._add_fault("MOMENTUM_SWING", 10, "Don't lean back!")
                
            # FAULT: Shrugging
            # If current shoulders are higher (smaller Y) than target_depth (start shoulders)
            if avg_sh_y < (self.target_depth_y - 0.05):
                self._add_fault("SHRUGGING", 5, "Keep shoulders down!")

            # TRANSITION: Velocity flips negative (starts going back up)
            if self.velocity < -0.1:
                # CHECK PARTIAL REP HERE (At the turnaround point)
                rom_percentage = self.max_wrist_y / self.target_depth_y
                
                if rom_percentage < 0.50:
                    self.feedback_buffer = "Too shallow"
                    self.state = "RETURNING" 
                    self.ignore_rep = True # Don't count, don't fail
                elif rom_percentage < 0.85:
                    self._add_fault("PARTIAL_REP", 5, "Pull lower to chest!")
                    self.state = "RETURNING"
                else:
                    self.feedback_buffer = "Good squeeze!"
                    self.state = "RETURNING"

        # C. RETURNING (Eccentric - Bar moving up)
        elif self.state == "RETURNING":
            if "PARTIAL_REP" not in [f['code'] for f in self.faults] and self.current_score > 50:
                self.feedback_buffer = "Control the weight up"
                
            # TRANSITION: Back to top stretch position
            if avg_wr_y < (self.target_depth_y * 0.15):
                self.state = "COMPLETE"

        # D. COMPLETE
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
            "faults": list(set([f['code'] for f in self.faults]))
        }

    # --- HELPERS ---
    def _start_rep(self, torso_len, target_y):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_wrist_y = 0.0
        self.start_torso_length = torso_len
        self.target_depth_y = target_y # Shoulder level at start
        self.ignore_rep = False

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finish_rep(self):
        # If marked to ignore (e.g. < 50% ROM), skip everything
        if self.ignore_rep:
            return

        if self.current_score > 50:
            self.rep_count += 1
        else:
            self.feedback_buffer = "Rep Failed (Bad Form)"