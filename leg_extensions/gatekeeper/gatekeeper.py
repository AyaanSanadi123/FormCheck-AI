import numpy as np
import time

class Gatekeeper:
    def __init__(self):
        # --- Configuration ---
        self.CONSISTENCY_WINDOW = 45  # Number of frames to hold position
        self.REQUIRED_VISIBILITY = 0.85
        
        # --- State Management ---
        self.window_buffer = []
        self.feedback = "Sit in profile. Keep your back against the seat."

    def check(self, landmarks):
        """
        Main interface method as per Blueprint standards.
        Returns: (passed: bool, message: str, calibration_data: dict)
        """
        if not landmarks or len(landmarks) < 33:
            return False, "User not detected", {}

        # 1. Visibility Check (Hip: 24, Knee: 26, Ankle: 28, Foot: 32)
        # We also check Shoulder (12) for torso orientation.
        critical_indices = [12, 24, 26, 28, 32]
        for idx in critical_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.window_buffer = []
                return False, "Ensure your side profile is fully visible", {}

        # 2. Determine Active Side (Facing Camera)
        # Higher visibility on right hip/knee suggests right-side profile
        active_side = "RIGHT" if landmarks[24].visibility > landmarks[23].visibility else "LEFT"
        
        # Joint mapping based on active side
        h_idx, k_idx, a_idx = (24, 26, 28) if active_side == "RIGHT" else (23, 25, 27)

        # 3. Profile Orientation Check (Z-Depth)
        # Ensures user is standing/sitting at 90 degrees to camera
        z_depth = abs(landmarks[11].z - landmarks[12].z)
        if z_depth < 0.20:
            self.window_buffer = []
            return False, "Rotate 90 degrees to show a side profile", {}

        # 4. Starting State Check (Legs Tucked)
        # Legs must be at < 110 degrees flexion to start the set
        current_flexion = self._calculate_angle(landmarks[h_idx], landmarks[k_idx], landmarks[a_idx])
        if current_flexion > 110:
            self.window_buffer = []
            return False, "Lower the weight to starting position (tucked)", {}

        # 5. Stability & Calibration Accumulation
        # Latch Thigh Length as Scale Factor (Hip to Knee)
        thigh_len = self._dist(landmarks[h_idx], landmarks[k_idx])
        
        self.window_buffer.append({
            'scale_factor': thigh_len,
            'knee_y': landmarks[k_idx].y,
            'hip_y': landmarks[h_idx].y
        })

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            calibration_data = self._finalize_calibration(active_side)
            return True, "Calibration Successful. Begin!", calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        return False, f"Hold position... {progress}%", {}

    def _dist(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))

    def _finalize_calibration(self, active_side):
        return {
            'active_side': active_side,
            'scale_factor': np.mean([f['scale_factor'] for f in self.window_buffer]),
            'knee_y_anchor': np.mean([f['knee_y'] for f in self.window_buffer]),
            'hip_y_anchor': np.mean([f['hip_y'] for f in self.window_buffer]),
            'exercise_id': 'seated-leg-extension',
            'timestamp': time.time()
        }