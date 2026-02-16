import numpy as np
import mediapipe as mp
import time
from collections import deque

class PushUpsGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.01
        self.SHOULDER_WIDTH_CHECK_RATIO = 0.5 # Shoulder width vs Torso length for side profile
        self.PLANK_ANGLE_THRESH = 160.0 # Degrees for shoulder-hip-ankle alignment
        
        # The Sliding Window
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # MediaPipe Indices
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        """
        Run per frame.
        Returns: (status: bool, message: str, calibration_data: dict/None)
        """
        if not landmarks:
            self._reset("No user detected")
            return False, "Looking for user...", None

        # Determine Active Side
        l_vis = (landmarks[self.MP_POSE.LEFT_SHOULDER.value].visibility + landmarks[self.MP_POSE.LEFT_HIP.value].visibility) / 2
        r_vis = (landmarks[self.MP_POSE.RIGHT_SHOULDER.value].visibility + landmarks[self.MP_POSE.RIGHT_HIP.value].visibility) / 2
        
        if l_vis > r_vis:
            active_side = 'LEFT'
            sh_idx, hip_idx, knee_idx, ank_idx, wr_idx = self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_HIP.value, self.MP_POSE.LEFT_KNEE.value, self.MP_POSE.LEFT_ANKLE.value, self.MP_POSE.LEFT_WRIST.value
        else:
            active_side = 'RIGHT'
            sh_idx, hip_idx, knee_idx, ank_idx, wr_idx = self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.RIGHT_KNEE.value, self.MP_POSE.RIGHT_ANKLE.value, self.MP_POSE.RIGHT_WRIST.value

        # --- STEP 1: VISIBILITY CHECK ---
        critical_pts = [sh_idx, hip_idx, knee_idx, ank_idx, wr_idx]
        for idx in critical_pts:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {self.MP_POSE(idx).name} hidden")
                return False, f"Ensure your {active_side} side is fully visible", None

        # --- STEP 2: ORIENTATION CHECK (Side Profile) ---
        shoulder_width = np.linalg.norm(
            np.array([landmarks[self.MP_POSE.LEFT_SHOULDER.value].x, landmarks[self.MP_POSE.LEFT_SHOULDER.value].y]) -
            np.array([landmarks[self.MP_POSE.RIGHT_SHOULDER.value].x, landmarks[self.MP_POSE.RIGHT_SHOULDER.value].y])
        )
        torso_len_estimate = np.linalg.norm(
            np.array([landmarks[sh_idx].x, landmarks[sh_idx].y]) -
            np.array([landmarks[hip_idx].x, landmarks[hip_idx].y])
        )
        
        if shoulder_width > (torso_len_estimate * self.SHOULDER_WIDTH_CHECK_RATIO):
            self._reset("Not a side profile")
            return False, "Please turn to a side-profile view", None

        # --- STEP 3: PLANK STABILITY (Initial Alignment) ---
        plank_angle = self._calculate_angle(landmarks[sh_idx], landmarks[hip_idx], landmarks[ank_idx])
        if plank_angle < self.PLANK_ANGLE_THRESH:
            self._reset("Not in plank position")
            return False, "Straighten your back and hips into a plank", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        frame_metrics = {
            "torso_length": torso_len_estimate,
            "floor_y_estimate": landmarks[wr_idx].y, # Wrist Y as approx floor
            "active_side": active_side,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Analyzing plank stability... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY! Start your push-ups!", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso length and floor estimate have been stable."""
        torso_len_history = [m['torso_length'] for m in self.validation_buffer]
        floor_y_history = [m['floor_y_estimate'] for m in self.validation_buffer]
        
        var_torso = np.var(torso_len_history)
        var_floor = np.var(floor_y_history)
        
        if var_torso > self.STABILITY_VARIANCE or var_floor > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_torso_len = np.mean([m['torso_length'] for m in self.validation_buffer])
        avg_floor_y = np.mean([m['floor_y_estimate'] for m in self.validation_buffer])
        active_side = self.validation_buffer[0]['active_side']
        
        return {
            "scale_factor": avg_torso_len,
            "floor_y_baseline": avg_floor_y,
            "active_side": active_side,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, p1, p2, p3):
        """Calculates angle between three points (p1-p2-p3)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))) if norm != 0 else 0
