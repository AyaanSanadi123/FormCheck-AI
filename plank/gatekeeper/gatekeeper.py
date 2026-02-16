import numpy as np
import mediapipe as mp
import time
from collections import deque

class PlankGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.90
        self.STABILITY_VARIANCE = 0.01
        self.STACKING_ERROR = 0.1
        self.ALIGNMENT_ERROR = 0.1
        
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
        l_vis = (landmarks[self.MP_POSE.LEFT_SHOULDER.value].visibility + landmarks[self.MP_POSE.LEFT_HIP.value].visibility + landmarks[self.MP_POSE.LEFT_ANKLE.value].visibility) / 3
        r_vis = (landmarks[self.MP_POSE.RIGHT_SHOULDER.value].visibility + landmarks[self.MP_POSE.RIGHT_HIP.value].visibility + landmarks[self.MP_POSE.RIGHT_ANKLE.value].visibility) / 3
        
        if l_vis > r_vis:
            active_side = 'LEFT'
            ear_idx, sh_idx, el_idx, hip_idx, ank_idx = self.MP_POSE.LEFT_EAR.value, self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_ELBOW.value, self.MP_POSE.LEFT_HIP.value, self.MP_POSE.LEFT_ANKLE.value
        else:
            active_side = 'RIGHT'
            ear_idx, sh_idx, el_idx, hip_idx, ank_idx = self.MP_POSE.RIGHT_EAR.value, self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_ELBOW.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.RIGHT_ANKLE.value

        # --- STEP 1: VISIBILITY CHECK ---
        critical_pts = [ear_idx, sh_idx, el_idx, hip_idx, ank_idx]
        for idx in critical_pts:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {self.MP_POSE(idx).name} hidden")
                return False, f"Ensure your {active_side} side is fully visible", None

        # --- STEP 2: JOINT STACKING CHECK (Elbow under Shoulder) ---
        stacking_error = abs(landmarks[sh_idx].x - landmarks[el_idx].x)
        if stacking_error > self.STACKING_ERROR:
            self._reset("Elbows not stacked")
            return False, "Place your elbows directly under your shoulders", None

        # --- STEP 3: INITIAL ALIGNMENT CHECK (Plank Form) ---
        sh_y, hip_y, ank_y = landmarks[sh_idx].y, landmarks[hip_idx].y, landmarks[ank_idx].y
        if not (min(sh_y, ank_y) - self.ALIGNMENT_ERROR < hip_y < max(sh_y, ank_y) + self.ALIGNMENT_ERROR):
            self._reset("Body not aligned")
            return False, "Align your hips with your shoulders and ankles", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        torso_len = np.linalg.norm(
            np.array([landmarks[sh_idx].x, landmarks[sh_idx].y]) -
            np.array([landmarks[hip_idx].x, landmarks[hip_idx].y])
        )

        frame_metrics = {
            "torso_length": torso_len,
            "active_side": active_side,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Analyzing structural integrity... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY! Hold the position!", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso length has been stable."""
        torso_len_history = [m['torso_length'] for m in self.validation_buffer]
        
        var_torso = np.var(torso_len_history)
        
        if var_torso > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_torso_len = np.mean([m['torso_length'] for m in self.validation_buffer])
        active_side = self.validation_buffer[0]['active_side'] # Get the consistent side
        
        return {
            "scale_factor": avg_torso_len,
            "active_side": active_side,
            "calibrated_at": time.time()
        }