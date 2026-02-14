import numpy as np
import mediapipe as mp
import time
from collections import deque

class HamstringCurlGatekeeper:
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5 
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.02 # Normalized units for standard deviation
        
        # Biomechanical Tolerances
        self.MAX_TORSO_ANGLE = 25.0  # Torso should be mostly flat (horizontal)
        self.MIN_LEG_ANGLE = 145.0   # Legs should be mostly extended to start
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        if not landmarks:
            self._reset()
            return False, "Looking for lifter...", None

        # --- STEP 1: VISIBILITY CHECK ---
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP,
            self.MP_POSE.LEFT_KNEE, self.MP_POSE.RIGHT_KNEE,
            self.MP_POSE.LEFT_ANKLE, self.MP_POSE.RIGHT_ANKLE
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, "Ensure Full Body Visible", None

        # --- STEP 2: CALCULATE JOINT AVERAGES ---
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        
        avg_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        avg_knee_x = (landmarks[25].x + landmarks[26].x) / 2
        avg_knee_y = (landmarks[25].y + landmarks[26].y) / 2
        
        avg_ankle_x = (landmarks[27].x + landmarks[28].x) / 2
        avg_ankle_y = (landmarks[27].y + landmarks[28].y) / 2

        # --- STEP 3: DETERMINE FACING DIRECTION ---
        # If shoulders are to the left of the hips, they are facing Right (Head on Left, Feet on Right)
        facing_dir = 1.0 if avg_sh_x < avg_hip_x else -1.0

        # --- STEP 4: THE PRONE CHECK (HORIZONTAL TORSO) ---
        dx = abs(avg_sh_x - avg_hip_x)
        dy = abs(avg_sh_y - avg_hip_y)
        if dx < 0.001: dx = 0.001
        
        # 0 degrees is perfectly horizontal
        torso_angle = np.degrees(np.arctan2(dy, dx))
        
        if torso_angle > self.MAX_TORSO_ANGLE:
            self._reset()
            return False, "Lie Flat on the Machine", None

        # --- STEP 5: THE EXTENSION CHECK (LEGS STRAIGHT) ---
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            
        leg_angle = self._calculate_angle(
            Point(avg_hip_x, avg_hip_y),
            Point(avg_knee_x, avg_knee_y),
            Point(avg_ankle_x, avg_ankle_y)
        )
        
        if leg_angle < self.MIN_LEG_ANGLE:
            self._reset()
            return False, "Lower the Weight (Legs Straight)", None

        # --- STEP 6: STABILITY BUFFER ---
        frame_metrics = {
            "ankle_x": avg_ankle_x,
            "ankle_y": avg_ankle_y,
            "knee_x": avg_knee_x,
            "knee_y": avg_knee_y,
            "hip_y": avg_hip_y,
            "leg_angle": leg_angle,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable():
            return False, "Hold Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, facing_dir)
        return True, "CURL READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the ankles have been still."""
        ankle_history = [m['ankle_x'] for m in self.validation_buffer]
        if np.std(ankle_history) > self.STABILITY_VARIANCE: 
            return False
        return True

    def _generate_passport(self, landmarks, facing_dir):
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
            "facing_side": facing_dir,
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