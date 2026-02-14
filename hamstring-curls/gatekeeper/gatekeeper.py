import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5 
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.02 
        
        # Biomechanical Tolerances
        self.MAX_TORSO_ANGLE = 25.0  # Torso should be mostly flat (horizontal)
        self.MIN_LEG_ANGLE = 145.0   # Legs should be mostly extended to start
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        if not landmarks:
            self._reset()
            return False, "Looking for lifter...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        # Compare Shoulder visibility
        left_vis = landmarks[11].visibility
        right_vis = landmarks[12].visibility
        active_side = "LEFT" if left_vis > right_vis else "RIGHT"

        # Select indices based on active side
        if active_side == "LEFT":
            idx_sh, idx_hip, idx_knee, idx_ankle = 11, 23, 25, 27
        else:
            idx_sh, idx_hip, idx_knee, idx_ankle = 12, 24, 26, 28

        # --- STEP 2: VISIBILITY CHECK (Active Side) ---
        critical_indices = [idx_sh, idx_hip, idx_knee, idx_ankle]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: THE PRONE CHECK (HORIZONTAL TORSO) ---
        sh_x, sh_y = landmarks[idx_sh].x, landmarks[idx_sh].y
        hip_x, hip_y = landmarks[idx_hip].x, landmarks[idx_hip].y
        
        dx = abs(sh_x - hip_x)
        dy = abs(sh_y - hip_y)
        if dx < 0.001: dx = 0.001
        
        # 0 degrees is perfectly horizontal
        torso_angle = np.degrees(np.arctan2(dy, dx))
        
        if torso_angle > self.MAX_TORSO_ANGLE:
            self._reset("Lie Flat")
            return False, "Lie Flat on the Machine", None

        # --- STEP 4: THE EXTENSION CHECK (LEGS STRAIGHT) ---
        knee_x, knee_y = landmarks[idx_knee].x, landmarks[idx_knee].y
        ankle_x, ankle_y = landmarks[idx_ankle].x, landmarks[idx_ankle].y
        
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            
        leg_angle = self._calculate_angle(
            Point(hip_x, hip_y),
            Point(knee_x, knee_y),
            Point(ankle_x, ankle_y)
        )
        
        if leg_angle < self.MIN_LEG_ANGLE:
            self._reset("Legs Straight")
            return False, "Lower the Weight (Legs Straight)", None

        # --- STEP 5: STABILITY BUFFER ---
        frame_metrics = {
            "active_side": active_side,
            "ankle_x": ankle_x,
            "ankle_y": ankle_y,
            "knee_x": knee_x,
            "knee_y": knee_y,
            "hip_y": hip_y,
            "leg_angle": leg_angle,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Hold Still...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, active_side)
        return True, "CURL READY!", calibration_data

    def _reset(self, reason=None):
        self.validation_buffer.clear()

    def _is_stable(self, current_side):
        """Checks if the ankles have been still."""
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False

        ankle_history = [m['ankle_x'] for m in self.validation_buffer]
        if np.std(ankle_history) > self.STABILITY_VARIANCE: 
            return False
        return True

    def _generate_passport(self, landmarks, active_side):
        """Captures user dimensions and setup state."""
        # 1. Knee Origin (To zero the Normalizer grid)
        knee_origin_x = np.mean([m['knee_x'] for m in self.validation_buffer])
        knee_origin_y = np.mean([m['knee_y'] for m in self.validation_buffer])
        
        # 2. Capture Baseline Hip Height (To catch 'Hip Lift' cheating)
        hip_baseline_y = np.mean([m['hip_y'] for m in self.validation_buffer])
        
        # 3. Leg Length (Scale factor: Knee to Ankle)
        avg_knee_x = np.mean([m['knee_x'] for m in self.validation_buffer])
        avg_knee_y = np.mean([m['knee_y'] for m in self.validation_buffer])
        avg_ankle_x = np.mean([m['ankle_x'] for m in self.validation_buffer])
        avg_ankle_y = np.mean([m['ankle_y'] for m in self.validation_buffer])
        
        leg_length = np.sqrt((avg_ankle_x - avg_knee_x)**2 + (avg_ankle_y - avg_knee_y)**2)
        
        return {
            "active_side": active_side,
            "knee_origin_x": knee_origin_x,
            "knee_origin_y": knee_origin_y,
            "hip_baseline_y": hip_baseline_y,
            "leg_length": leg_length,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))
