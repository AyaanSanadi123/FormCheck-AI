import numpy as np
import mediapipe as mp
import time
from collections import deque

class HangingLegRaisesGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.01
        self.NECK_GAP_THRESH = 0.04
        self.PLUMB_LINE_TOLERANCE = 0.15
        
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

        # Determine Lead Side
        l_vis = (landmarks[self.MP_POSE.LEFT_SHOULDER.value].visibility + landmarks[self.MP_POSE.LEFT_HIP.value].visibility + landmarks[self.MP_POSE.LEFT_ANKLE.value].visibility) / 3
        r_vis = (landmarks[self.MP_POSE.RIGHT_SHOULDER.value].visibility + landmarks[self.MP_POSE.RIGHT_HIP.value].visibility + landmarks[self.MP_POSE.RIGHT_ANKLE.value].visibility) / 3
        
        if l_vis > r_vis:
            active_side = 'LEFT'
            sh_idx, wr_idx, hip_idx, ank_idx, ear_idx = self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_WRIST.value, self.MP_POSE.LEFT_HIP.value, self.MP_POSE.LEFT_ANKLE.value, self.MP_POSE.LEFT_EAR.value
        else:
            active_side = 'RIGHT'
            sh_idx, wr_idx, hip_idx, ank_idx, ear_idx = self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_WRIST.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.RIGHT_ANKLE.value, self.MP_POSE.RIGHT_EAR.value

        # --- STEP 1: VISIBILITY CHECK ---
        critical_pts = [sh_idx, wr_idx, hip_idx, ank_idx, ear_idx]
        for idx in critical_pts:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {self.MP_POSE(idx).name} hidden")
                return False, f"Ensure your {active_side} side is fully visible", None

        # --- STEP 2: ACTIVE HANG CHECK ---
        neck_gap = abs(landmarks[sh_idx].y - landmarks[ear_idx].y)
        if neck_gap < self.NECK_GAP_THRESH:
            self._reset("Shoulders not engaged")
            return False, "Engage your shoulders (depress scapulae)", None

        # --- STEP 3: PLUMB LINE CHECK (DEAD HANG) ---
        plumb_line_err = abs(landmarks[wr_idx].x - landmarks[hip_idx].x)
        if plumb_line_err > self.PLUMB_LINE_TOLERANCE:
            self._reset("Not a dead hang")
            return False, "Stop swinging to calibrate", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        torso_len = np.linalg.norm(
            np.array([landmarks[sh_idx].x, landmarks[sh_idx].y]) -
            np.array([landmarks[hip_idx].x, landmarks[hip_idx].y])
        )

        frame_metrics = {
            "torso_length": torso_len,
            "bar_x": landmarks[wr_idx].x,
            "bar_y": landmarks[wr_idx].y,
            "active_side": active_side,
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
        return True, "SYSTEM READY! Raise your legs!", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso length and bar position has been stable."""
        torso_len_history = [m['torso_length'] for m in self.validation_buffer]
        bar_x_history = [m['bar_x'] for m in self.validation_buffer]
        
        var_torso = np.var(torso_len_history)
        var_bar_x = np.var(bar_x_history)
        
        if var_torso > self.STABILITY_VARIANCE or var_bar_x > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_torso_len = np.mean([m['torso_length'] for m in self.validation_buffer])
        avg_bar_x = np.mean([m['bar_x'] for m in self.validation_buffer])
        avg_bar_y = np.mean([m['bar_y'] for m in self.validation_buffer])
        active_side = self.validation_buffer[0]['active_side'] # Get the consistent side
        
        return {
            "scale_factor": avg_torso_len,
            "bar_origin": (avg_bar_x, avg_bar_y),
            "active_side": active_side,
            "calibrated_at": time.time()
        }
