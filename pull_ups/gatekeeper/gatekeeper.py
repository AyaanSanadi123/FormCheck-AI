import numpy as np
import mediapipe as mp
import time
from collections import deque

class PullUpsGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.01
        self.HANGING_Y_THRESHOLD = -0.05 # Wrists must be above shoulders
        self.SHOULDER_LEVEL_TOLERANCE = 0.08
        
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

        # --- STEP 1: VISIBILITY CHECK ---
        critical_indices = [
            self.MP_POSE.LEFT_EAR, self.MP_POSE.RIGHT_EAR,
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_ELBOW, self.MP_POSE.RIGHT_ELBOW,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {self.MP_POSE(idx.value).name} hidden")
                return False, "Ensure full upper body and bar are visible", None

        # --- STEP 2: GEOMETRY - DEAD HANG CHECK ---
        # Wrists must be significantly higher than shoulders (negative Y in normalized space)
        # Assuming person is facing camera or side view, not directly underneath bar
        l_wrist_y = landmarks[self.MP_POSE.LEFT_WRIST.value].y
        r_wrist_y = landmarks[self.MP_POSE.RIGHT_WRIST.value].y
        l_sh_y = landmarks[self.MP_POSE.LEFT_SHOULDER.value].y
        r_sh_y = landmarks[self.MP_POSE.RIGHT_SHOULDER.value].y

        if not (l_wrist_y < l_sh_y + self.HANGING_Y_THRESHOLD and r_wrist_y < r_sh_y + self.HANGING_Y_THRESHOLD):
            self._reset("Not a dead hang")
            return False, "Start from a dead hang position", None

        # --- STEP 3: SYMMETRY CHECK ---
        shoulder_level = abs(landmarks[self.MP_POSE.LEFT_SHOULDER.value].y - landmarks[self.MP_POSE.RIGHT_SHOULDER.value].y)
        if shoulder_level > self.SHOULDER_LEVEL_TOLERANCE:
            self._reset("Shoulders not level")
            return False, "Level your shoulders with the bar", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        torso_len = np.linalg.norm(
            np.array([landmarks[self.MP_POSE.LEFT_SHOULDER.value].x, landmarks[self.MP_POSE.LEFT_SHOULDER.value].y]) -
            np.array([landmarks[self.MP_POSE.LEFT_HIP.value].x, landmarks[self.MP_POSE.LEFT_HIP.value].y])
        )
        bar_y_estimate = (landmarks[self.MP_POSE.LEFT_WRIST.value].y + landmarks[self.MP_POSE.RIGHT_WRIST.value].y) / 2

        frame_metrics = {
            "torso_length": torso_len,
            "bar_y_estimate": bar_y_estimate,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Analyzing hang stability... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY! Pull up until your chin clears the bar.", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso length and bar position has been stable."""
        torso_len_history = [m['torso_length'] for m in self.validation_buffer]
        bar_y_history = [m['bar_y_estimate'] for m in self.validation_buffer]
        
        var_torso = np.var(torso_len_history)
        var_bar_y = np.var(bar_y_history)
        
        if var_torso > self.STABILITY_VARIANCE or var_bar_y > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_torso_len = np.mean([m['torso_length'] for m in self.validation_buffer])
        avg_bar_y = np.mean([m['bar_y_estimate'] for m in self.validation_buffer])
        
        return {
            "scale_factor": avg_torso_len,
            "bar_y_baseline": avg_bar_y,
            "active_side": "RIGHT", # Default to RIGHT for frontal/two-sided exercise
            "calibrated_at": time.time()
        }
