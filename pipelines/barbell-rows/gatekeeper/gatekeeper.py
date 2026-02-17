import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.70
        self.STABILITY_VARIANCE = 0.005 # For Torso/Hips
        
        # Biomechanical Tolerances
        self.MAX_TORSO_ANGLE = 60.0  # Degrees from horizontal (0 = flat back)
        self.MIN_TORSO_ANGLE = 10.0
        self.MIN_KNEE_ANGLE = 120.0  # Squatting too deep
        self.MAX_KNEE_ANGLE = 175.0  # Stiff-legged
        self.ARM_VERTICAL_TOLERANCE = 0.15 # Max horizontal drift for dead hang
        
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
            return False, "Looking for lifter...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        # Compare visibility of Left vs Right Hips
        left_vis = landmarks[self.MP_POSE.LEFT_HIP.value].visibility
        right_vis = landmarks[self.MP_POSE.RIGHT_HIP.value].visibility
        active_side = "LEFT" if left_vis > right_vis else "RIGHT"

        # Select indices based on active side
        if active_side == "LEFT":
            idx_sh, idx_hip, idx_knee, idx_ankle, idx_wr = 11, 23, 25, 27, 15
        else:
            idx_sh, idx_hip, idx_knee, idx_ankle, idx_wr = 12, 24, 26, 28, 16

        # --- STEP 2: VISIBILITY CHECK (Active Side) ---
        critical_indices = [idx_sh, idx_hip, idx_knee, idx_ankle, idx_wr]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx} hidden")
                return False, f"{active_side.title()} Side Visible?", None

        # --- STEP 3: THE HINGE CHECK (TORSO ANGLE) ---
        # Calculate angle of torso relative to horizontal plane
        # MediaPipe +Y is down. Hips should be lower or equal to shoulders.
        sh_x, sh_y = landmarks[idx_sh].x, landmarks[idx_sh].y
        hip_x, hip_y = landmarks[idx_hip].x, landmarks[idx_hip].y
        
        dx = abs(hip_x - sh_x)
        dy = abs(hip_y - sh_y)
        
        # If standing straight, dx = 0, angle = 90. If flat back, dy = 0, angle = 0.
        torso_angle = np.degrees(np.arctan2(dy, dx))
        
        if torso_angle > self.MAX_TORSO_ANGLE:
            self._reset("Too Upright")
            return False, "Bend Over More (Hinge)", None
        
        if torso_angle < self.MIN_TORSO_ANGLE:
            self._reset("Too Flat")
            return False, "Chest Up Slightly", None

        # --- STEP 4: THE ANCHOR CHECK (KNEE FLEXION) ---
        # Calculate Knee Angle (Hip -> Knee -> Ankle)
        knee_x, knee_y = landmarks[idx_knee].x, landmarks[idx_knee].y
        ankle_x, ankle_y = landmarks[idx_ankle].x, landmarks[idx_ankle].y
        
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            
        knee_angle = self._calculate_angle(
            Point(hip_x, hip_y),
            Point(knee_x, knee_y),
            Point(ankle_x, ankle_y)
        )
        
        if knee_angle > self.MAX_KNEE_ANGLE:
            self._reset("Legs Straight")
            return False, "Slightly Bend Knees", None
        if knee_angle < self.MIN_KNEE_ANGLE:
            self._reset("Squatting")
            return False, "Don't Squat (Raise Hips)", None

        # --- STEP 5: ARM VERTICALITY (DEAD HANG) ---
        # Active Wrist should be directly below Active Shoulder.
        wr_x, wr_y = landmarks[idx_wr].x, landmarks[idx_wr].y
        
        if abs(wr_x - sh_x) > self.ARM_VERTICAL_TOLERANCE:
            self._reset("Arms Angled")
            return False, "Arms Straight Down", None
            
        # Ensure wrists are below shoulders
        if wr_y < sh_y:
            self._reset("Arms Up")
            return False, "Lower the Bar", None

        # --- STEP 6: STABILITY BUFFER ---
        frame_metrics = {
            "active_side": active_side,
            "torso_angle": torso_angle,
            "ankle_y": ankle_y,
            "sh_y": sh_y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable(active_side):
            return False, "Hold Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, active_side)
        return True, "ROW READY!", calibration_data

    def _reset(self, reason=None):
        self.validation_buffer.clear()

    def _is_stable(self, current_side):
        """Checks if the torso angle has been still."""
        if self.validation_buffer[0]['active_side'] != current_side:
            self.validation_buffer.clear()
            return False

        angle_history = [m['torso_angle'] for m in self.validation_buffer]
        if np.var(angle_history) > 2.0: # Variance in degrees
            return False
        return True

    def _generate_passport(self, landmarks, active_side):
        """Captures user dimensions and setup state."""
        # 1. Calculate Floor Level (Average Ankle Y)
        floor_y = np.mean([m['ankle_y'] for m in self.validation_buffer])
        
        # 2. Capture Setup Torso Angle
        setup_torso_angle = np.mean([m['torso_angle'] for m in self.validation_buffer])
        
        # 3. Torso Length (Scale factor)
        if active_side == "LEFT":
            sh = landmarks[11]; hip = landmarks[23]
        else:
            sh = landmarks[12]; hip = landmarks[24]
            
        torso_length = np.sqrt((sh.x - hip.x)**2 + (sh.y - hip.y)**2)
        
        return {
            "active_side": active_side,
            "floor_y": floor_y,
            "setup_torso_angle": setup_torso_angle,
            "torso_length": torso_length,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))
