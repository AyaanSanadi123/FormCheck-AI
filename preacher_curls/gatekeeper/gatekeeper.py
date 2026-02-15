import numpy as np
import time

class Gatekeeper:
    def __init__(self):
        """
        Initializes Preacher Curl Gatekeeper.
        Standard: 45-frame stability window for calibration.
        """
        self.CONSISTENCY_WINDOW = 45  
        self.REQUIRED_VISIBILITY = 0.85
        self.window_buffer = []
        self.feedback = "Sit at the bench and extend your arm fully."

    def check(self, landmarks):
        """
        Verifies profile orientation and bench setup.
        Returns: (passed: bool, message: str, calibration_data: dict)
        """
        if not landmarks or len(landmarks) < 33:
            return False, "Searching for user...", {}

        # 1. Visibility Check: Shoulder (12), Elbow (14), Wrist (16)
        # Preacher curls focus on the upper extremity.
        active_side = "RIGHT" if landmarks[12].visibility > landmarks[11].visibility else "LEFT"
        s_idx, e_idx, w_idx = (12, 14, 16) if active_side == "RIGHT" else (11, 13, 15)
        
        critical_indices = [s_idx, e_idx, w_idx]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.window_buffer = []
                return False, "Ensure your arm and shoulder are fully visible", {}

        # 2. Side Profile Check (Z-Depth)
        # Preacher curls must be analyzed from the side to see elbow flexion.
        z_depth = abs(landmarks[11].z - landmarks[12].z)
        if z_depth < 0.20:
            self.window_buffer = []
            return False, "Rotate 90 degrees to show a side profile", {}

        # 3. Initial State: Arm Extension
        # User should start at the bottom of the bench to calibrate max extension.
        start_angle = self._calculate_angle(landmarks[s_idx], landmarks[e_idx], landmarks[w_idx])
        if start_angle < 150:
            self.window_buffer = []
            return False, "Extend your arm fully on the pad to begin", {}

        # 4. Stability & Calibration
        # scale_factor = Upper Arm Length (Shoulder to Elbow)
        humerus_len = self._dist(landmarks[s_idx], landmarks[e_idx])
        
        self.window_buffer.append({
            'scale_factor': humerus_len,
            'elbow_y_baseline': landmarks[e_idx].y,
            'bench_angle': start_angle # Angle of the pad relative to torso
        })

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            calibration_data = self._finalize_calibration(active_side)
            return True, "Calibration Successful. Start Curling!", calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        return False, f"Calibrating bench angle... {progress}%", {}

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
            'elbow_y_anchor': np.mean([f['elbow_y_baseline'] for f in self.window_buffer]),
            'exercise_id': 'preacher-curl',
            'timestamp': time.time()
        }