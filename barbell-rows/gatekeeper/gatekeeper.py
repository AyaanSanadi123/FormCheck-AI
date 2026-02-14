import numpy as np
import mediapipe as mp
import time
from collections import deque

class BarbellRowGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.005 # For Torso/Hips
        
        # Biomechanical Tolerances
        self.MAX_TORSO_ANGLE = 60.0  # Degrees from horizontal (0 = flat back, 90 = standing upright)
        self.MIN_TORSO_ANGLE = 10.0
        self.MIN_KNEE_ANGLE = 120.0  # Squatting too deep
        self.MAX_KNEE_ANGLE = 175.0  # Stiff-legged
        self.ARM_VERTICAL_TOLERANCE = 0.15 # Max horizontal drift for dead hang
        
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
            self._reset()
            return False, "Looking for lifter...", None

        # --- STEP 1: VISIBILITY CHECK ---
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP,
            self.MP_POSE.LEFT_KNEE, self.MP_POSE.RIGHT_KNEE,
            self.MP_POSE.LEFT_ANKLE, self.MP_POSE.RIGHT_ANKLE,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, "Full Body Visible?", None

        # --- STEP 2: CALCULATE JOINT AVERAGES ---
        # Averages provide robustness regardless of facing direction
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        
        avg_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        avg_knee_x = (landmarks[25].x + landmarks[26].x) / 2
        avg_knee_y = (landmarks[25].y + landmarks[26].y) / 2
        
        avg_ankle_x = (landmarks[27].x + landmarks[28].x) / 2
        avg_ankle_y = (landmarks[27].y + landmarks[28].y) / 2
        
        avg_wrist_x = (landmarks[15].x + landmarks[16].x) / 2
        avg_wrist_y = (landmarks[15].y + landmarks[16].y) / 2

        # --- STEP 3: THE HINGE CHECK (TORSO ANGLE) ---
        # Calculate angle of torso relative to horizontal plane
        dx = abs(avg_hip_x - avg_sh_x)
        dy = abs(avg_hip_y - avg_sh_y) # MediaPipe +Y is down. Hips should be lower or equal to shoulders.
        
        # If standing straight, dx = 0, angle = 90. If flat back, dy = 0, angle = 0.
        torso_angle = np.degrees(np.arctan2(dy, dx))
        
        if torso_angle > self.MAX_TORSO_ANGLE:
            self._reset()
            return False, "Bend Over More (Hinge)", None
        
        if torso_angle < self.MIN_TORSO_ANGLE:
            self._reset()
            return False, "Chest Up Slightly", None

        # --- STEP 4: THE ANCHOR CHECK (KNEE FLEXION) ---
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            
        knee_angle = self._calculate_angle(
            Point(avg_hip_x, avg_hip_y),
            Point(avg_knee_x, avg_knee_y),
            Point(avg_ankle_x, avg_ankle_y)
        )
        
        if knee_angle > self.MAX_KNEE_ANGLE:
            self._reset()
            return False, "Slightly Bend Knees", None
        if knee_angle < self.MIN_KNEE_ANGLE:
            self._reset()
            return False, "Don't Squat (Raise Hips)", None

        # --- STEP 5: ARM VERTICALITY (DEAD HANG) ---
        # Wrists should be directly below shoulders.
        if abs(avg_wrist_x - avg_sh_x) > self.ARM_VERTICAL_TOLERANCE:
            self._reset()
            return False, "Arms Straight Down", None
            
        # Ensure wrists are below shoulders (not already pulling)
        if avg_wrist_y < avg_sh_y:
            self._reset()
            return False, "Lower the Bar", None

        # --- STEP 6: STABILITY BUFFER ---
        frame_metrics = {
            "torso_angle": torso_angle,
            "ankle_y": avg_ankle_y,
            "sh_y": avg_sh_y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable():
            return False, "Hold Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks, avg_hip_x)
        return True, "ROW READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the torso angle has been still."""
        angle_history = [m['torso_angle'] for m in self.validation_buffer]
        if np.var(angle_history) > 2.0: # Variance in degrees
            return False
        return True

    def _generate_passport(self, landmarks, hip_center_x):
        """Captures user dimensions and setup state."""
        # 1. Calculate Floor Level (Average Ankle Y)
        floor_y = np.mean([m['ankle_y'] for m in self.validation_buffer])
        
        # 2. Capture Setup Torso Angle (For the 'Heave' fault later)
        setup_torso_angle = np.mean([m['torso_angle'] for m in self.validation_buffer])
        
        # 3. Determine Facing Side (-1 for Left, 1 for Right)
        nose_x = landmarks[0].x
        facing_side = -1 if nose_x < hip_center_x else 1
        
        # 4. Torso Length (Scale factor)
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        torso_length = np.sqrt((avg_sh_x - hip_center_x)**2 + (avg_sh_y - avg_hip_y)**2)
        
        return {
            "floor_y": floor_y,
            "setup_torso_angle": setup_torso_angle,
            "facing_side": facing_side,
            "torso_length": torso_length,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))