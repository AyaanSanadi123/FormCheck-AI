import numpy as np
import time

class Gatekeeper:
    def __init__(self):
        """
        Initializes the Leg Press Gatekeeper.
        Standard: 45-frame stability window for calibration.
        """
        self.CONSISTENCY_WINDOW = 45  
        self.REQUIRED_VISIBILITY = 0.85
        self.window_buffer = []
        self.feedback = "Position yourself in the Leg Press machine (Side View)."

    def check(self, landmarks):
        """
        Verifies machine alignment and user orientation.
        Returns: (passed: bool, message: str, calibration_data: dict)
        """
        if not landmarks or len(landmarks) < 33:
            return False, "Searching for user...", {}

        # 1. Critical Visibility Check
        # Hip(24), Knee(26), Ankle(28) are vital for the leg press lever.
        critical_indices = [12, 24, 26, 28] 
        for idx in critical_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.window_buffer = []
                return False, "Leg or hip obscured. Adjust camera to side view.", {}

        # 2. Side Profile Check (Z-Depth)
        # Leg press must be analyzed in the sagittal plane.
        z_depth = abs(landmarks[11].z - landmarks[12].z)
        if z_depth < 0.20:
            self.window_buffer = []
            return False, "Rotate camera to a side-profile view of the machine.", {}

        # 3. Determine Active Side
        active_side = "RIGHT" if landmarks[24].visibility > landmarks[23].visibility else "LEFT"
        h_idx, k_idx, a_idx = (24, 26, 28) if active_side == "RIGHT" else (23, 25, 27)

        # 4. Starting Geometry Check (Extended Legs)
        # Most leg presses start with legs extended to unlock the safety.
        start_angle = self._calculate_angle(landmarks[h_idx], landmarks[k_idx], landmarks[a_idx])
        if start_angle < 140:
            self.window_buffer = []
            return False, "Extend your legs fully to start calibration.", {}

        # 5. Stability & Calibration Accumulation
        # Scale Factor: Torso length (Shoulder to Hip)
        s_idx = 12 if active_side == "RIGHT" else 11
        torso_len = self._dist(landmarks[s_idx], landmarks[h_idx])
        
        self.window_buffer.append({
            'scale_factor': torso_len,
            'hip_y_baseline': landmarks[h_idx].y,
            'knee_y_baseline': landmarks[k_idx].y
        })

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            calibration_data = self._finalize_calibration(active_side)
            return True, "Calibration Successful. Start your reps!", calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        return False, f"Hold position... {progress}%", {}

    def _dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm == 0: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))

    def _finalize_calibration(self, active_side):
        return {
            'active_side': active_side,
            'scale_factor': np.mean([f['scale_factor'] for f in self.window_buffer]),
            'hip_y_anchor': np.mean([f['hip_y_baseline'] for f in self.window_buffer]),
            'knee_y_anchor': np.mean([f['knee_y_baseline'] for f in self.window_buffer]),
            'exercise_id': 'leg-press',
            'timestamp': time.time()
        }