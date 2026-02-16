import numpy as np
import time

class StandingLateralRaiseGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.CONSISTENCY_WINDOW = 45  # ~1.5 seconds at 30 FPS
        self.REQUIRED_VISIBILITY = 0.90
        
        # State Management
        self.is_calibrated = False
        self.calibration_data = {}
        self.window_buffer = []
        
        # Feedback for UI
        self.feedback = "Stand facing camera with dumbbells at sides"

    def check(self, landmarks):
        """
        Validates starting posture and latches anatomical baselines.
        Key Addition: Torso Scale (Shoulder-to-Hip) for shrugging detection.
        """
        if not landmarks or len(landmarks) < 33:
            return False, "No landmarks detected", None

        # 1. Essential Joints for Frontal Proportions
        # 11,12=Shoulders | 23,24=Hips | 15,16=Wrists
        essential_indices = [11, 12, 13, 14, 15, 16, 23, 24]
        
        # 2. Frame-Level Rejection: Visibility
        for idx in essential_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.feedback = "Ensure your full body is visible"
                self.window_buffer = []
                return False, self.feedback, None

        # 3. Frame-Level Rejection: Perspective (Z-Axis Symmetry)
        # Ensuring the user isn't rotated before we latch torso proportions
        sh_z_diff = abs(landmarks[11].z - landmarks[12].z)
        if sh_z_diff > 0.15: 
            self.feedback = "Face the camera directly"
            self.window_buffer = []
            return False, self.feedback, None

        # 4. Frame-Level Rejection: Initial Position
        # Must start with weights below hips
        avg_wrist_y = (landmarks[15].y + landmarks[16].y) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        if avg_wrist_y < avg_hip_y:
            self.feedback = "Lower weights to your sides"
            self.window_buffer = []
            return False, self.feedback, None

        # 5. Stability Check & Anatomical Metric Gathering
        current_metrics = {
            'torso_l': self._calculate_torso_scale(landmarks),
            'arm_l': self._calculate_arm_scale(landmarks),
            'sh_y': (landmarks[11].y + landmarks[12].y) / 2,
            'hip_x': (landmarks[23].x + landmarks[24].x) / 2
        }
        self.window_buffer.append(current_metrics)

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            self._finalize_calibration()
            return True, self.feedback, self.calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        self.feedback = f"Analyzing Posture... {progress}%"
        return False, self.feedback, None

    def _calculate_torso_scale(self, lm):
        """
        Calculates the Euclidean distance between shoulders and hips.
        This provides the baseline length of the spine.
        """
        # Left Side: Dist(Shoulder_11, Hip_23)
        l_torso = np.linalg.norm([lm[11].x - lm[23].x, lm[11].y - lm[23].y])
        # Right Side: Dist(Shoulder_12, Hip_24)
        r_torso = np.linalg.norm([lm[12].x - lm[24].x, lm[12].y - lm[24].y])
        return (l_torso + r_torso) / 2

    def _calculate_arm_scale(self, lm):
        """Calculates total arm length (Upper + Lower) for ROM scaling."""
        l_arm = (np.linalg.norm([lm[11].x - lm[13].x, lm[11].y - lm[13].y]) + 
                 np.linalg.norm([lm[13].x - lm[15].x, lm[13].y - lm[15].y]))
        r_arm = (np.linalg.norm([lm[12].x - lm[14].x, lm[12].y - lm[14].y]) + 
                 np.linalg.norm([lm[14].x - lm[16].x, lm[14].y - lm[16].y]))
        return (l_arm + r_arm) / 2

    def _finalize_calibration(self):
        """Averages the buffer to create the Calibration Passport."""
        self.calibration_data = {
            'active_side': "RIGHT", # Changed from "BOTH" to "RIGHT" for blueprint compliance
            'scale_factor': np.mean([f['torso_l'] for f in self.window_buffer]), # Torso length as scale factor
            'arm_length': np.mean([f['arm_l'] for f in self.window_buffer]),
            'sh_y_baseline': np.mean([f['sh_y'] for f in self.window_buffer]),
            'neutral_hip_x': np.mean([f['hip_x'] for f in self.window_buffer]),
            'exercise_id': 'standing_lateral_raise',
            'calibrated_at': time.time()
        }
        self.is_calibrated = True
        self.feedback = "Calibration Complete! Start Raising."