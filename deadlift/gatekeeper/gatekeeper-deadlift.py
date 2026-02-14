import numpy as np
import mediapipe as mp
import time
from collections import deque

class DeadliftGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.70
        self.STABILITY_VARIANCE = 0.005 
        
        # The Sliding Window
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        """
        Run per frame.
        Returns: (status: bool, message: str, calibration_data: dict/None)
        """
        if not landmarks:
            self._reset("No user detected")
            return False, "Looking for lifter...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        # Compare visibility of Left vs Right Hips
        left_vis = landmarks[self.MP_POSE.LEFT_HIP.value].visibility
        right_vis = landmarks[self.MP_POSE.RIGHT_HIP.value].visibility
        active_side = "LEFT" if left_vis > right_vis else "RIGHT"

        # Select indices based on active side
        if active_side == "LEFT":
            idx_sh, idx_hip, idx_knee, idx_ankle = 11, 23, 25, 27
            idx_wr = 15
        else:
            idx_sh, idx_hip, idx_knee, idx_ankle = 12, 24, 26, 28
            idx_wr = 16

        # --- STEP 2: VISIBILITY CHECK (Active Side) ---
        # We need the full chain on the active side
        critical_indices = [idx_sh, idx_hip, idx_knee, idx_ankle, idx_wr]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx} hidden")
                return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: TRIANGLE HIERARCHY CHECK ---
        # Rule: Shoulders > Hips > Knees (In height/elevation)
        # MediaPipe Y: 0 is Top, 1 is Bottom.
        
        sh_y = landmarks[idx_sh].y
        hip_y = landmarks[idx_hip].y
        knee_y = landmarks[idx_knee].y
        
        # Check 1: Are Hips below Shoulders?
        if sh_y >= hip_y:
            self._reset("Hips too high")
            return False, "Lower your Hips", None
            
        # Check 2: Are Knees below Hips?
        if hip_y >= knee_y:
            self._reset("Squatting too deep")
            return False, "Raise Hips (Don't Squat)", None

        # --- STEP 4: SHIN VERTICALITY CHECK ---
        # Active Knee vs Active Ankle X
        knee_x = landmarks[idx_knee].x
        ankle_x = landmarks[idx_ankle].x
        
        if abs(knee_x - ankle_x) > 0.10: 
            self._reset("Shins too angled")
            return False, "Shins Vertical (Hips Back)", None

        # --- STEP 5: ARM VERTICALITY CHECK ---
        # Active Shoulder vs Active Wrist X
        sh_x = landmarks[idx_sh].x
        wr_x = landmarks[idx_wr].x
        
        if abs(sh_x - wr_x) > 0.15:
            self._reset("Arms not vertical")
            return False, "Arms Straight Down", None

        # --- STEP 6: STABILITY BUFFER ---
        frame_metrics = {
            "active_side": active_side,
            "hip_y": hip_y,
            "ankle_y": landmarks[idx_ankle].y, # Floor height
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Stay Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, active_side)
        return True, "DEADLIFT READY!", calibration_data

    def _reset(self, reason):
        self.validation_buffer.clear()

    def _is_stable(self, current_side):
        """Checks if Hips have been still."""
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False

        hip_history = [m['hip_y'] for m in self.validation_buffer]
        if np.var(hip_history) > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, landmarks, active_side):
        """Captures user dimensions."""
        # Calculate Floor Level (Average Ankle Y of active side)
        floor_y = np.mean([m['ankle_y'] for m in self.validation_buffer])
        
        if active_side == "LEFT":
            sh = landmarks[11]; hip = landmarks[23]
        else:
            sh = landmarks[12]; hip = landmarks[24]
        
        # Calculate Torso Length (Shoulder to Hip)
        torso_length = np.sqrt((sh.x - hip.x)**2 + (sh.y - hip.y)**2)
        
        return {
            "active_side": active_side,
            "floor_y": floor_y,
            "torso_length": torso_length,
            "calibrated_at": time.time()
        }
