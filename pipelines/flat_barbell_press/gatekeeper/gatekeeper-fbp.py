import numpy as np
import mediapipe as mp
import time
from collections import deque

class BenchGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.005 # Wrist stability (Bar must be still)
        self.HORIZONTAL_TOLERANCE = 0.20 # Max Y-diff between Shoulder & Hip
        self.ANGLE_TOLERANCE = (60, 120) # Side View Only
        
        # The Sliding Window
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        if not landmarks:
            self._reset("No user detected")
            return False, "Looking for lifter...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        # Compare visibility of Left vs Right Shoulders
        left_sh_vis = landmarks[11].visibility
        right_sh_vis = landmarks[12].visibility
        active_side = "LEFT" if left_sh_vis > right_sh_vis else "RIGHT"

        if active_side == "LEFT":
            idx_sh, idx_hip, idx_el, idx_wr = 11, 23, 13, 15
        else:
            idx_sh, idx_hip, idx_el, idx_wr = 12, 24, 14, 16

        # --- STEP 2: VISIBILITY CHECK (Active Side) ---
        critical_indices = [idx_sh, idx_hip, idx_el, idx_wr]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: HORIZONTAL ORIENTATION CHECK ---
        # User must be lying down.
        sh_y = landmarks[idx_sh].y
        hip_y = landmarks[idx_hip].y
        
        if abs(sh_y - hip_y) > self.HORIZONTAL_TOLERANCE:
            self._reset("User not flat")
            return False, "Lie flat on the bench", None

        # --- STEP 4: ANGLE CHECK ---
        angle = self._calculate_facing_angle(landmarks)
        if not (self.ANGLE_TOLERANCE[0] < abs(angle) < self.ANGLE_TOLERANCE[1]):
            self._reset("Bad Angle")
            return False, "Camera must be Side-On", None

        # --- STEP 5: STABILITY BUFFER ---
        wr_y = landmarks[idx_wr].y
        frame_metrics = {
            "active_side": active_side,
            "wr_y": wr_y,
            "sh_y": sh_y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Bar Steady... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Don't move the bar...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, active_side)
        return True, "BENCH PRESS READY!", calibration_data

    def _reset(self, reason=None):
        self.validation_buffer.clear()

    def _calculate_facing_angle(self, landmarks):
        l_hip = landmarks[23]; r_hip = landmarks[24]
        dx = l_hip.x - r_hip.x
        dz = l_hip.z - r_hip.z
        return np.degrees(np.arctan2(dz, dx))

    def _is_stable(self, current_side):
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False
        bar_history = [m['wr_y'] for m in self.validation_buffer]
        if np.var(bar_history) > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, landmarks, active_side):
        avg_bench_height = np.mean([m['sh_y'] for m in self.validation_buffer])
        
        if active_side == "LEFT":
            sh = landmarks[11]; wr = landmarks[15]
            facing_side = -1.0 # Typically facing Left if Left side is visible
        else:
            sh = landmarks[12]; wr = landmarks[16]
            facing_side = 1.0
        
        # Arm length (Shoulder to Wrist)
        arm_length = np.sqrt((sh.x - wr.x)**2 + (sh.y - wr.y)**2)
        
        return {
            "active_side": active_side,
            "facing_side": facing_side,
            "bench_y": avg_bench_height,
            "arm_length": arm_length,
            "calibrated_at": time.time()
        }
