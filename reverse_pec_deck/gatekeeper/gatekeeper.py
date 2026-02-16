import numpy as np
import mediapipe as mp
import time
from collections import deque

class ReversePecDeckGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.01
        self.VERTICAL_SPINE_TOLERANCE = 0.1
        self.SHOULDER_LEVEL_TOLERANCE = 0.05
        
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
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_ELBOW, self.MP_POSE.RIGHT_ELBOW,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx.name} hidden")
                return False, "Ensure entire upper body is visible", None

        # --- STEP 2: POSTURE CHECK ---
        mid_sh_x = (landmarks[self.MP_POSE.LEFT_SHOULDER.value].x + landmarks[self.MP_POSE.RIGHT_SHOULDER.value].x) / 2
        mid_hip_x = (landmarks[self.MP_POSE.LEFT_HIP.value].x + landmarks[self.MP_POSE.RIGHT_HIP.value].x) / 2
        
        if abs(mid_sh_x - mid_hip_x) > self.VERTICAL_SPINE_TOLERANCE:
            self._reset("Spine not vertical")
            return False, "Sit upright, align back vertically", None

        shoulder_level = abs(landmarks[self.MP_POSE.LEFT_SHOULDER.value].y - landmarks[self.MP_POSE.RIGHT_SHOULDER.value].y)
        if shoulder_level > self.SHOULDER_LEVEL_TOLERANCE:
            self._reset("Shoulders not level")
            return False, "Level your shoulders", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        shoulder_width = np.linalg.norm(
            np.array([landmarks[self.MP_POSE.LEFT_SHOULDER.value].x, landmarks[self.MP_POSE.LEFT_SHOULDER.value].y]) -
            np.array([landmarks[self.MP_POSE.RIGHT_SHOULDER.value].x, landmarks[self.MP_POSE.RIGHT_SHOULDER.value].y])
        )

        frame_metrics = {
            "shoulder_width": shoulder_width,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Still... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY!", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the shoulder width has been stable."""
        shoulder_width_history = [m['shoulder_width'] for m in self.validation_buffer]
        
        var_shoulder_width = np.var(shoulder_width_history)
        
        if var_shoulder_width > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_scale = np.mean([m['shoulder_width'] for m in self.validation_buffer])
        
        return {
            "scale_factor": avg_scale,
            "calibrated_at": time.time()
        }
