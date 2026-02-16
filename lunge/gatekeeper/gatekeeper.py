import numpy as np
import mediapipe as mp
import time
from collections import deque

class LungeGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.01
        self.ALIGNMENT_TOLERANCE = 0.2
        
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

        # Determine Lead Leg (Side facing camera)
        l_vis = (landmarks[self.MP_POSE.LEFT_HIP.value].visibility + landmarks[self.MP_POSE.LEFT_KNEE.value].visibility + landmarks[self.MP_POSE.LEFT_ANKLE.value].visibility) / 3
        r_vis = (landmarks[self.MP_POSE.RIGHT_HIP.value].visibility + landmarks[self.MP_POSE.RIGHT_KNEE.value].visibility + landmarks[self.MP_POSE.RIGHT_ANKLE.value].visibility) / 3
        
        if l_vis > r_vis:
            active_side = 'LEFT'
            sh_idx, hip_idx, knee_idx, ank_idx, toe_idx = self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_HIP.value, self.MP_POSE.LEFT_KNEE.value, self.MP_POSE.LEFT_ANKLE.value, self.MP_POSE.LEFT_FOOT_INDEX.value
        else:
            active_side = 'RIGHT'
            sh_idx, hip_idx, knee_idx, ank_idx, toe_idx = self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.RIGHT_KNEE.value, self.MP_POSE.RIGHT_ANKLE.value, self.MP_POSE.RIGHT_FOOT_INDEX.value

        # --- STEP 1: VISIBILITY CHECK ---
        critical_pts = [sh_idx, hip_idx, knee_idx, ank_idx, toe_idx]
        for idx in critical_pts:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {self.MP_POSE(idx).name} hidden")
                return False, f"Ensure your {active_side} side is fully visible", None

        # --- STEP 2: BASELINE STANCE CHECK ---
        alignment_err = abs(landmarks[hip_idx].x - landmarks[ank_idx].x)
        if alignment_err > self.ALIGNMENT_TOLERANCE:
            self._reset("Stance not aligned")
            return False, "Start from a stable, upright stance", None

        # --- STEP 3: TEMPORAL BUFFERING ---
        torso_len = np.linalg.norm(
            np.array([landmarks[sh_idx].x, landmarks[sh_idx].y]) -
            np.array([landmarks[hip_idx].x, landmarks[hip_idx].y])
        )

        frame_metrics = {
            "torso_length": torso_len,
            "floor_y": landmarks[ank_idx].y,
            "active_side": active_side,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 4: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Calibrating stance... {progress}%", None

        if not self._is_stable():
            return False, "Don't move...", None

        # --- STEP 5: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "SYSTEM READY! Drop into the lunge.", calibration_data

    def _reset(self, reason):
        """Clears the buffer."""
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso length and floor position has been stable."""
        torso_len_history = [m['torso_length'] for m in self.validation_buffer]
        floor_y_history = [m['floor_y'] for m in self.validation_buffer]
        
        var_torso = np.var(torso_len_history)
        var_floor = np.var(floor_y_history)
        
        if var_torso > self.STABILITY_VARIANCE or var_floor > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_torso_len = np.mean([m['torso_length'] for m in self.validation_buffer])
        avg_floor_y = np.mean([m['floor_y'] for m in self.validation_buffer])
        active_side = self.validation_buffer[0]['active_side'] # Get the consistent side
        
        return {
            "scale_factor": avg_torso_len,
            "floor_y_baseline": avg_floor_y,
            "active_side": active_side,
            "calibrated_at": time.time()
        }
