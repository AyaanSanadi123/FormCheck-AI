import numpy as np
import time
from collections import deque

class OneArmRowRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.setup_torso_angle = calibration_data.get('setup_torso_angle', 15.0)
        
        # State Management (Up-First)
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Pull to your hip."
        
        # Physics Tracking
        self.prev_wr_y = None
        self.prev_time = None
        self.wrist_velocity_y = 0.0
        self.velocity_history = deque(maxlen=5) 
        
        # Rep Bounds (Metrics for fault checking)
        self.start_wrist_x = 0.0
        self.max_elbow_y = -999.0
        self.min_wrist_x = 999.0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT ACTIVE JOINTS ---
        # Hip is (0,0). User faces RIGHT (+X). Up is +Y.
        if self.active_side == "LEFT":
            sh = landmarks[11]; hip = landmarks[23]
            el = landmarks[13]; wr = landmarks[15]
        else:
            sh = landmarks[12]; hip = landmarks[24]
            el = landmarks[14]; wr = landmarks[16]

        # --- STEP 2: CALCULATE METRICS ---
        curr_time = timestamp if timestamp else time.time()
        
        # Torso Angle (Live)
        dx_torso = sh.x - hip.x
        dy_torso = sh.y - hip.y
        if dx_torso < 0.001: dx_torso = 0.001
        current_torso_angle = np.degrees(np.arctan2(dy_torso, dx_torso))
        
        # Elbow Angle (for extension checks)
        elbow_angle = self._calculate_angle(sh, el, wr)

        if self.prev_wr_y is None:
            self.prev_wr_y = wr.y
            self.prev_time = curr_time
            return None 

        # Smoothed Wrist Y-Velocity (Positive = Pulling UP, Negative = Dropping DOWN)
        if self.prev_time is None or curr_time == self.prev_time:
            dt = 1.0 / self.FPS
        else:
            dt = curr_time - self.prev_time
            
        if dt <= 0: dt = 0.001
            
        raw_vel = (wr.y - self.prev_wr_y) / dt
        self.velocity_history.append(raw_vel)
        self.wrist_velocity_y = np.mean(self.velocity_history) if self.velocity_history else 0.0
        
        self.prev_wr_y = wr.y
        self.prev_time = curr_time

        # --- STEP 3: STATE MACHINE (Up-First) ---

        # A. IDLE (Dead Hang)
        if self.state == "IDLE":
            self.feedback_buffer = "Pull to your hip"
            
            # TRIGGER: Wrist moves up rapidly
            if self.wrist_velocity_y > 0.5 and elbow_angle < 160.0:
                self._start_rep(wr.x)
                self.state = "CONCENTRIC"

        # B. CONCENTRIC (Pulling weight UP and BACK)
        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Squeeze the lat!"
            
            # Track peak metrics
            if el.y > self.max_elbow_y: self.max_elbow_y = el.y
            if wr.x < self.min_wrist_x: self.min_wrist_x = wr.x

            # SEVERE FAULT: Torso Heave (Lawnmower Cheat)
            if current_torso_angle > self.setup_torso_angle + 15.0:
                self._add_fault("TORSO_HEAVE", 10, "Don't jerk your back! Stay flat.")

            # TRANSITION: Wrist velocity flips negative (dropping)
            if self.wrist_velocity_y < -0.2:
                
                # MINOR FAULT: Short Pull
                # Elbow should reach roughly the height of the shoulder
                if self.max_elbow_y < sh.y - 0.10:
                    self._add_fault("SHORT_PULL", 5, "Pull your elbow higher!")
                    
                # MINOR FAULT: Bicep Pull
                # Wrist must travel backwards (-X) to form the J-Curve
                travel_x = self.start_wrist_x - self.min_wrist_x
                if travel_x < 0.10:
                    self._add_fault("BICEP_PULL", 5, "Pull back to your hip, not your chest!")

                self.state = "ECCENTRIC"

        # C. ECCENTRIC (Lowering weight DOWN)
        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control the weight down"
            
            # SEVERE FAULT: Torso Heave (sometimes happens on the drop to catch it)
            if current_torso_angle > self.setup_torso_angle + 15.0:
                self._add_fault("TORSO_HEAVE", 10, "Keep your back stable on the way down!")
            
            # TRANSITION: Arm is fully extended again (Dead Hang)
            if elbow_angle > 150.0:
                self._finish_rep(success=True)
                self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
        # Strictly compliant with PIPELINE_BLUEPRINT.md
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "coords": landmarks,          
            "raw_coords": raw_landmarks,  
            "faults": list(set([f['code'] for f in self.faults])),
            "metrics": {
                "torso_angle": int(current_torso_angle),
                "elbow_angle": int(elbow_angle),
                "wrist_vel_y": round(self.wrist_velocity_y, 2),
                "active_side": self.active_side
            }
        }

    # --- HELPERS ---
    def _start_rep(self, start_wr_x):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.start_wrist_x = start_wr_x
        self.max_elbow_y = -999.0
        self.min_wrist_x = 999.0

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