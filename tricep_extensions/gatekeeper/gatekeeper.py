import numpy as np
import time

class Gatekeeper:
    def __init__(self):
        """
        Initializes the Tricep Extension Gatekeeper.
        Standard: 45-frame stability window for calibration.
        """
        self.CONSISTENCY_WINDOW = 45  
        self.REQUIRED_VISIBILITY = 0.85
        self.window_buffer = []
        self.feedback = "Stand sideways to the camera. Hold the bar at chest height."

    def check(self, landmarks):
        """
        Verifies profile orientation and arm positioning.
        Returns: (passed: bool, message: str, calibration_data: dict)
        """
        if not landmarks or len(landmarks) < 33:
            return False, "Searching for user...", {}

        # 1. Visibility Check: Shoulder (12), Elbow (14), Wrist (16)
        # We need the full arm lever to track the extension accurately.
        active_side = "RIGHT" if landmarks[12].visibility > landmarks[11].visibility else "LEFT"
        s_idx, e_idx, w_idx = (12, 14, 16) if active_side == "RIGHT" else (11, 13, 15)
        h_idx = 24 if active_side == "RIGHT" else 23 # Hip for torso alignment

        critical_indices = [s_idx, e_idx, w_idx, h_idx]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.window_buffer = []
                return False, "Ensure your arm and torso are fully visible", {}

        # 2. Side Profile Check (Z-Depth)
        # Sagittal plane view is required to measure elbow flexion/extension.
        z_depth = abs(landmarks[11].z - landmarks[12].z)
        if z_depth < 0.20:
            self.window_buffer = []
            return False, "Please turn 90 degrees for a side profile view.", {}

        # 3. Initial Geometry: Torso Verticality
        # Prevents calibration if the user is leaning too far into the machine.
        torso_angle = self._calculate_vertical_angle(landmarks[s_idx], landmarks[h_idx])
        if torso_angle > 30: # Max 30 degree lean allowed
            self.window_buffer = []
            return False, "Stand up straighter to calibrate.", {}

        # 4. Stability & Calibration
        # scale_factor = Humerus length (Shoulder to Elbow)
        humerus_len = self._dist(landmarks[s_idx], landmarks[e_idx])
        
        self.window_buffer.append({
            'scale_factor': humerus_len,
            'elbow_y_baseline': landmarks[e_idx].y,
            'shoulder_x_baseline': landmarks[s_idx].x
        })

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            calibration_data = self._finalize_calibration(active_side)
            return True, "Calibration Successful. Start the pushdown!", calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        return False, f"Calibrating arm position... {progress}%", {}

    def _dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _calculate_vertical_angle(self, p_top, p_bottom):
        """Calculates deviation from vertical line (0 degrees is straight up/down)."""
        dx = abs(p_top.x - p_bottom.x)
        dy = abs(p_top.y - p_bottom.y)
        return np.degrees(np.arctan2(dx, dy))

    def _finalize_calibration(self, active_side):
        return {
            'active_side': active_side,
            'scale_factor': np.mean([f['scale_factor'] for f in self.window_buffer]),
            'elbow_y_anchor': np.mean([f['elbow_y_baseline'] for f in self.window_buffer]),
            'shoulder_x_anchor': np.mean([f['shoulder_x_baseline'] for f in self.window_buffer]),
            'exercise_id': 'tricep-extension',
            'timestamp': time.time()
        }