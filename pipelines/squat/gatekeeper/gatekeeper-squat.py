import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.75 # Lowered slightly for side view realism
        self.STABILITY_VARIANCE = 0.015 
        
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
            return False, "Looking for a human...", None

        # --- STEP 1: DETERMINE FACING SIDE & ACTIVE LIMBS ---
        # We check which hip is more visible to determine the active side.
        left_hip_vis = landmarks[self.MP_POSE.LEFT_HIP.value].visibility
        right_hip_vis = landmarks[self.MP_POSE.RIGHT_HIP.value].visibility
        
        active_side = "LEFT" if left_hip_vis > right_hip_vis else "RIGHT"
        
        # Select indices based on active side
        if active_side == "LEFT":
            idx_sh, idx_hip, idx_knee, idx_ankle = 11, 23, 25, 27
            idx_heel = 29
        else:
            idx_sh, idx_hip, idx_knee, idx_ankle = 12, 24, 26, 28
            idx_heel = 30

        # --- STEP 2: VISIBILITY CHECK (Active Side Only) ---
        # We only require the active side to be visible.
        # Nose is always required.
        nose_vis = landmarks[0].visibility
        sh_vis = landmarks[idx_sh].visibility
        hip_vis = landmarks[idx_hip].visibility
        knee_vis = landmarks[idx_knee].visibility
        ankle_vis = landmarks[idx_ankle].visibility
        
        if (nose_vis < self.VISIBILITY_THRESH or 
            sh_vis < self.VISIBILITY_THRESH or 
            hip_vis < self.VISIBILITY_THRESH or
            knee_vis < self.VISIBILITY_THRESH or
            ankle_vis < self.VISIBILITY_THRESH):
            
            self._reset("Active side occluded")
            return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: INTEGRITY CHECKS ---
        # A. Edge Clipping (Active Side)
        points_to_check = [0, idx_sh, idx_hip, idx_knee, idx_ankle]
        for idx in points_to_check:
            lm = landmarks[idx]
            if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
                 self._reset("User clipping edge")
                 return False, "Step closer to center.", None

        # B. Distance (Scale) Check
        # Torso length (Shoulder to Hip)
        torso_len = abs(landmarks[idx_hip].y - landmarks[idx_sh].y)
        if torso_len < 0.15:
            return False, "Move Closer.", None
        if torso_len > 0.7:
            return False, "Step Back.", None

        # --- STEP 4: BUFFERING ---
        frame_metrics = {
            "active_side": active_side,
            "hip_x": landmarks[idx_hip].x,
            "hip_y": landmarks[idx_hip].y,
            "heel_y": landmarks[idx_heel].y,
            "torso_len": torso_len,
            # Angle approx: Atan2 of Hip-Shoulder vector
            "angle": np.degrees(np.arctan2(landmarks[idx_hip].y - landmarks[idx_sh].y, 
                                           landmarks[idx_hip].x - landmarks[idx_sh].x)),
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Still... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Don't move...", None

        # --- STEP 6: CALIBRATION PASSPORT ---
        calibration_data = self._generate_passport(active_side)
        return True, "SQUAT READY!", calibration_data

    def _reset(self, reason):
        self.validation_buffer.clear()

    def _is_stable(self, current_side):
        """Checks variance of Hip X/Y."""
        # Ensure we haven't switched sides in the middle of calibration
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False

        hip_x_history = [m['hip_x'] for m in self.validation_buffer]
        hip_y_history = [m['hip_y'] for m in self.validation_buffer]
        
        if np.std(hip_x_history) > self.STABILITY_VARIANCE or np.std(hip_y_history) > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, active_side):
        """Averages the buffer to create a robust calibration profile."""
        avg_floor = np.mean([m['heel_y'] for m in self.validation_buffer])
        avg_torso = np.mean([m['torso_len'] for m in self.validation_buffer])
        
        # Determine Facing Direction
        # Side-view check: Nose X vs Hip X
        # We need landmarks for this... but we can infer from the last frame buffer logic or just pass active_side
        # To be precise, we trust the Normalizer will handle the rotation if we give it the side.
        
        return {
            "active_side": active_side,
            "floor_y": avg_floor,
            "calibrated_scale": avg_torso,
            "calibrated_at": time.time()
        }
