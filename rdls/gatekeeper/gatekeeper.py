import numpy as np
import mediapipe as mp
import time
from collections import deque

class RdlsGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.90
        self.STABILITY_VARIANCE = 0.01
        self.HIP_ANKLE_STACK_TOLERANCE = 0.1
        
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
            return False, "Looking for lifter...", None

        # Determine Active Side
        l_vis = (landmarks[self.MP_POSE.LEFT_HIP.value].visibility + landmarks[self.MP_POSE.LEFT_KNEE.value].visibility + landmarks[self.MP_POSE.LEFT_ANKLE.value].visibility) / 3
        r_vis = (landmarks[self.MP_POSE.RIGHT_HIP.value].visibility + landmarks[self.MP_POSE.RIGHT_KNEE.value].visibility + landmarks[self.MP_POSE.RIGHT_ANKLE.value].visibility) / 3
        
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

        # --- STEP 2: INITIAL ALIGNMENT CHECK (Standing Tall) ---
        hip_x_norm = landmarks[hip_idx].x
        ankle_x_norm = landmarks[ank_idx].x
        if abs(hip_x_norm - ankle_x_norm) > self.HIP_ANKLE_STACK_TOLERANCE:
            self._reset("Hips not stacked over ankles")
            return False, "Stand upright with hips over ankles to begin", None

        # --- STEP 3: TEMPORAL BUFFERING ---
        leg_len = np.linalg.norm(
            np.array([landmarks[hip_idx].x, landmarks[hip_idx].y]) -
            np.array([landmarks[ank_idx].x, landmarks[ank_idx].y])
        )
        base_knee_angle = self._calculate_angle(landmarks[hip_idx], landmarks[knee_idx], landmarks[ank_idx])

        frame_metrics = {
            "leg_length": leg_len,
            "base_knee_angle": base_knee_angle,
            "active_side": active_side,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 4: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Calibrating side profile... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 5: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY! Begin the hinge.", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the leg length and knee angle have been stable."""
        leg_len_history = [m['leg_length'] for m in self.validation_buffer]
        knee_angle_history = [m['base_knee_angle'] for m in self.validation_buffer]
        
        var_leg_len = np.var(leg_len_history)
        var_knee_angle = np.var(knee_angle_history)
        
        if var_leg_len > self.STABILITY_VARIANCE or var_knee_angle > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_leg_len = np.mean([m['leg_length'] for m in self.validation_buffer])
        avg_base_knee_angle = np.mean([m['base_knee_angle'] for m in self.validation_buffer])
        active_side = self.validation_buffer[0]['active_side']
        
        return {
            "scale_factor": avg_leg_len,
            "base_knee_angle": avg_base_knee_angle,
            "active_side": active_side,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, p1, p2, p3):
        """Calculates angle between three points (p1-p2-p3)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))) if norm != 0 else 0
