import numpy as np
from collections import deque

class HamstringCurlRep:
    def __init__(self, calibration_data):
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.active_side = calibration_data.get('active_side', "RIGHT")
        self.hip_baseline_y = calibration_data.get('hip_baseline_y', 0.0)
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Legs straight, prepare to curl."
        
        # Physics Tracking (Angular Tracking)
        self.prev_ank_angle = None 
        self.angular_velocity = 0.0
        self.velocity_history = deque(maxlen=5) # Smooth over 5 frames
        
        # Rep Bounds
        self.start_angle = 0.0
        self.max_curl_angle = 0.0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT JOINTS (Normalized & Active) ---
        if self.active_side == "LEFT":
            hip = landmarks[23]; knee = landmarks[25]; ank = landmarks[27]
        else:
            hip = landmarks[24]; knee = landmarks[26]; ank = landmarks[28]
        
        hip_y = hip.y
        ank_x = ank.x
        ank_y = ank.y

        # Prevent division by zero or exactly 0/0 scenarios
        if ank_x == 0 and ank_y == 0: ank_x = 0.001

        # --- STEP 2: CALCULATE METRICS ---
        # Calculate angle of the ankle relative to the knee (0,0)
        # 0 degrees = straight out to the right. 90 degrees = straight up.
        current_angle = np.degrees(np.arctan2(ank_y, ank_x))
        
        if self.prev_ank_angle is None:
            self.prev_ank_angle = current_angle
            return None # Skip first frame velocity spike

        # Smoothed Angular Velocity (Degrees per second)
        dt = 1.0 / self.FPS
        raw_vel = (current_angle - self.prev_ank_angle) / dt
        self.velocity_history.append(raw_vel)
        self.angular_velocity = np.mean(self.velocity_history) if self.velocity_history else 0.0
        self.prev_ank_angle = current_angle

        # --- STEP 3: STATE MACHINE ---

        # A. IDLE (Legs Extended)
        if self.state == "IDLE":
            self.feedback_buffer = "Curl heels to glutes"
            
            # TRIGGER: Angular velocity is positive (moving UP/curling)
            if self.angular_velocity > 15.0 and current_angle > (self.start_angle + 5.0):
                self._start_rep(current_angle)
                self.state = "CURLING"

        # B. CURLING (Concentric - Pad moving UP/BACK)
        elif self.state == "CURLING":
            self.feedback_buffer = "Squeeze hamstrings!"
            
            # Track maximum curl angle
            if current_angle > self.max_curl_angle:
                self.max_curl_angle = current_angle

            # FAULT: Hip Lift (Cheating with lower back/momentum)
            # If hips rise more than 15% of a leg length above baseline
            # (Note: Normalized Y means UP is positive. Baseline is raw... Wait)
            
            # CRITICAL FIX: hip_baseline_y from Gatekeeper is RAW Y (0=Top).
            # But Normalizer transforms Y to be (0=Knee, +Y=Up).
            # We cannot compare Normalized Hip Y to Raw Baseline Hip Y.
            # We must track Hip Lift relative to Normalized Knee (0).
            # If Normalized Hip Y rises significantly, it means Hips are going UP relative to knee.
            
            # Since user is prone, Knee and Hip should be roughly same Y (0).
            # If Hip Y becomes > 0.15, they are lifting hips.
            if hip_y > 0.15:
                self._add_fault("HIP_LIFT", 10, "Keep hips pressed flat into the pad!")

            # TRANSITION: Angular velocity flips negative (starts going down)
            # OR if angle drops significantly below max (slow eccentric)
            if self.angular_velocity < -5.0 or (current_angle < self.max_curl_angle - 10.0):
                # CHECK SHORT CURL HERE
                # Full curl is usually 85+ degrees. 
                if self.max_curl_angle < 80.0:
                    self._add_fault("SHORT_CURL", 5, "Curl all the way up!")
                
                self.state = "RETURNING"

        # C. RETURNING (Eccentric - Pad moving DOWN)
        elif self.state == "RETURNING":
            self.feedback_buffer = "Control the weight down"
            
            # TRANSITION: Pad returns to near starting extension (e.g., < 25 degrees)
            if current_angle <= 25.0:
                self._finish_rep(success=True)
                self.state = "IDLE"
                # Reset baseline but clamp it to avoid drift creep
                self.start_angle = min(current_angle, 25.0) 
                
            # EARLY REVERSAL (Bouncing the weight)
            elif self.angular_velocity > 15.0:
                self._add_fault("NO_STRETCH", 5, "Let legs extend fully between reps!")
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
            "curl_angle": int(current_angle)
        }

    # --- HELPERS ---
    def _start_rep(self, start_ang):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_curl_angle = start_ang

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