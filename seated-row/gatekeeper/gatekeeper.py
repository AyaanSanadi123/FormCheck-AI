import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.0 
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.75
        self.STABILITY_VARIANCE = 0.01 
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        if not landmarks:
            self._reset()
            return False, "Looking for rower...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        # Compare Shoulder visibility
        left_vis = landmarks[11].visibility
        right_vis = landmarks[12].visibility
        active_side = "LEFT" if left_vis > right_vis else "RIGHT"

        if active_side == "LEFT":
            idx_sh, idx_hip, idx_knee, idx_wr = 11, 23, 25, 15
        else:
            idx_sh, idx_hip, idx_knee, idx_wr = 12, 24, 26, 16

        # --- STEP 2: VISIBILITY CHECK ---
        for idx in [idx_sh, idx_hip, idx_knee, idx_wr]:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: POSTURE CHECK (Sitting Tall) ---
        # Shoulder should be roughly above Hip (small X difference)
        sh_x = landmarks[idx_sh].x
        hip_x = landmarks[idx_hip].x
        
        # Check Torso Angle (approx vertical)
        # Using raw X/Y to check vertical alignment
        if abs(sh_x - hip_x) > 0.2:
            self._reset()
            return False, "Sit Up Straight", None

        # --- STEP 4: STABILITY BUFFER ---
        frame_metrics = {
            "active_side": active_side,
            "sh_x": sh_x,
            "sh_y": landmarks[idx_sh].y,
            "hip_x": hip_x,
            "hip_y": landmarks[idx_hip].y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Start... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Hold Still...", None

        # --- STEP 5: CALIBRATION ---
        calibration_data = self._generate_passport(active_side)
        return True, "ROW READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self, current_side):
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False
            
        sh_x_hist = [m['sh_x'] for m in self.validation_buffer]
        if np.std(sh_x_hist) > self.STABILITY_VARIANCE: 
            return False
        return True

    def _generate_passport(self, active_side):
        # Calculate Setup Torso Angle
        avg_dx = np.mean([m['sh_x'] - m['hip_x'] for m in self.validation_buffer])
        avg_dy = np.mean([m['sh_y'] - m['hip_y'] for m in self.validation_buffer])
        
        # Angle from vertical? Or horizontal? Rep logic uses "Angle from Horizontal"
        # Hip is origin. Dy is height. Dx is run.
        # MediaPipe Y increases DOWN. So Hip Y > Shoulder Y.
        # dy should be negative (Shoulder - Hip).
        # We want angle relative to vertical or horizontal.
        
        # Rep logic uses: np.degrees(np.arctan2(dy, dx))
        # Let's pass the raw average angle
        angle = np.degrees(np.arctan2(avg_dy, avg_dx))
        if angle < 0: angle += 360
        
        return {
            "active_side": active_side,
            "setup_torso_angle": angle,
            "calibrated_at": time.time()
        }
