import numpy as np
import time

class Gatekeeper:
    def __init__(self):
        """
        Initializes the Calf Raise Gatekeeper.
        Standard: Stability buffer of 45 frames.
        """
        self.CONSISTENCY_WINDOW = 45  
        self.REQUIRED_VISIBILITY = 0.85
        self.window_buffer = []
        self.feedback = "Stand sideways. Keep your feet flat to begin."

    def check(self, landmarks):
        """
        Verifies side-profile visibility and stability before passing.
        Returns: (passed: bool, message: str, calibration_data: dict).
        """
        if not landmarks or len(landmarks) < 33:
            return False, "Searching for user...", {}

        # 1. Critical Visibility Check (Hip: 24, Knee: 26, Ankle: 28, Heel: 30, Toe: 32)
        # Calf raises require high precision on foot/heel landmarks.
        critical_indices = [12, 24, 26, 28, 30, 32]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.window_buffer = []
                return False, "Ensure your legs and feet are fully visible", {}

        # 2. Determine Active Side
        # Higher visibility on right hip/knee suggests right-side profile.
        active_side = "RIGHT" if landmarks[24].visibility > landmarks[23].visibility else "LEFT"
        h_idx, k_idx, a_idx, heel_idx, toe_idx = (24, 26, 28, 30, 32) if active_side == "RIGHT" else (23, 25, 27, 29, 31)

        # 3. Profile Orientation Check (Z-Depth)
        # Side profile is mandatory to observe heel elevation.
        z_depth = abs(landmarks[11].z - landmarks[12].z)
        if z_depth < 0.20:
            self.window_buffer = []
            return False, "Rotate 90 degrees to show a side profile", {}

        # 4. Starting Geometry: "Feet Flat" Verification
        # Heel and toe should be on approximately the same Y-plane initially.
        heel_y = landmarks[heel_idx].y
        toe_y = landmarks[toe_idx].y
        if abs(heel_y - toe_y) > 0.05:
            self.window_buffer = []
            return False, "Stand with heels flat on the floor", {}

        # 5. Stability & Calibration Accumulation
        # scale_factor = Torso length (Shoulder to Hip).
        torso_len = self._dist(landmarks[12 if active_side == "RIGHT" else 11], landmarks[h_idx])
        
        self.window_buffer.append({
            'scale_factor': torso_len,
            'floor_y': max(heel_y, toe_y),
            'heel_y_baseline': heel_y
        })

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            calibration_data = self._finalize_calibration(active_side)
            return True, "Calibration Successful. Start raising heels!", calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        return False, f"Holding steady... {progress}%", {}

    def _dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _finalize_calibration(self, active_side):
        """
        Aggregates buffer into the calibration dictionary.
        """
        return {
            'active_side': active_side,
            'scale_factor': np.mean([f['scale_factor'] for f in self.window_buffer]),
            'floor_y': np.mean([f['floor_y'] for f in self.window_buffer]),
            'heel_y_baseline': np.mean([f['heel_y_baseline'] for f in self.window_buffer]),
            'exercise_id': 'calf-raise',
            'timestamp': time.time()
        }