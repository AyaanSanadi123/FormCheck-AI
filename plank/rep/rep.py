import numpy as np
import time

class PlankRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # User Baselines
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.active_side = calibration_data.get('active_side', 'RIGHT')

        # Thresholds (Normalized units based on Torso Length)
        self.THRESH_SAG = 0.15     # Hip Y deviation (Positive = Sagging)
        self.THRESH_PIKE = -0.15   # Hip Y deviation (Negative = Piking)
        self.THRESH_NECK = 0.10    # Ear Y deviation from shoulder line
        self.THRESH_STABILITY = 0.05 # Max Hip jitter/oscillation
        
        # State Management
        self.state = "IDLE" 
        self.hold_time = 0.0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Get into position"
        
        # Tracking
        self.prev_time = time.time()
        self.hip_buffer = deque(maxlen=15) # For jitter/fatigue analysis

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # Determine relevant indices for active side
        sh_idx = 12 if self.active_side == 'RIGHT' else 11
        hip_idx = 24 if self.active_side == 'RIGHT' else 23
        ank_idx = 28 if self.active_side == 'RIGHT' else 27
        ear_idx = 8 if self.active_side == 'RIGHT' else 7

        sh, hip, ank, ear = landmarks[sh_idx], landmarks[hip_idx], landmarks[ank_idx], landmarks[ear_idx]
        
        # --- STEP 2: CALCULATE METRICS & CHECK INTEGRITY ---
        is_aligned, fault_code, msg = self._check_integrity(sh, hip, ank, ear)
        
        # --- STEP 3: STATE MACHINE (Isometric Timer) ---
        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        if self.state == "IDLE":
            if is_aligned:
                self.state = "HOLDING"
                self.feedback_buffer = "Hold steady"
                self.hold_time = 0.0 # Start timer
                self.current_score = self.SCORE_MAX # Reset score
                self.faults = [] # Clear faults
            else:
                self.feedback_buffer = msg if msg else "Align your body"

        elif self.state == "HOLDING":
            if is_aligned:
                self.hold_time += dt
                self.feedback_buffer = "Core engaged"
                self._update_score(is_aligned=True)
            else:
                self.state = "FAILED"
                self.feedback_buffer = msg
                self._update_score(is_aligned=False, penalty=10)
                self._add_fault(fault_code, 10, msg) # Add the specific fault

        elif self.state == "FAILED":
            if is_aligned:
                self.state = "HOLDING"
                self.feedback_buffer = "Recovered. Keep holding!"
                self.current_score = max(0, self.current_score - 5) # Small penalty for recovery
            else:
                self.feedback_buffer = msg

        # --- STEP 4: PACKAGE OUTPUT ---
        return {
            "state": self.state,
            "reps": int(self.hold_time), # Reps = Seconds held
            "score": int(self.current_score),
            "feedback": self.feedback_buffer,
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"hold_duration": self.hold_time, "hip_y_norm": hip.y}
        }

    # --- HELPERS ---

    def _check_integrity(self, sh, hip, ank, ear):
        # A. Hip Sag (Gravity taking over)
        if hip.y > self.THRESH_SAG:
            return False, "HIP_SAG", "Brace your core - hips up!"
        
        # B. Piking (Relieving tension)
        if hip.y < self.THRESH_PIKE:
            return False, "HIP_PIKE", "Lower your hips to level"
            
        # C. Neck Neutrality
        if abs(ear.y - sh.y) > self.THRESH_NECK:
            return False, "NECK_DIVE", "Look at the floor, pack your neck"
            
        # D. Fatigue Jitter (Stability Score)
        self.hip_buffer.append(hip.y)
        if len(self.hip_buffer) == self.hip_buffer.maxlen:
            jitter = np.std(list(self.hip_buffer))
            if jitter > self.THRESH_STABILITY:
                return False, "STABILITY_LOSS", "Stop shaking - tighten everything"

        return True, None, None

    def _update_score(self, is_aligned, penalty=0):
        if is_aligned:
            # Gradually increase score if perfect, or maintain
            self.current_score = min(100, self.current_score + (0.5 / self.FPS)) # Small recovery per frame
        else:
            self.current_score = max(0, self.current_score - penalty)

    def _add_fault(self, code, penalty, msg):
        if code and not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg})
            self.feedback_buffer = msg
