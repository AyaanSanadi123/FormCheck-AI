import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class LatPullNormalizer:
    def __init__(self):
        # Calibration (from Gatekeeper)
        self.spine_center_x = 0.5 
        self.max_reach_y = 0.0 
        self.shoulder_width = 0.5
        
        # Smoothing for Rotation (prevents jitter)
        self.smoothed_angle = 0.0
        self.alpha = 0.1 # Soft smoothing factor

    @staticmethod
    def _get_val(lm, attr, default=0.0):
        if isinstance(lm, dict):
            return lm.get(attr, default)
        return getattr(lm, attr, default)

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks: Rotates to Flat View -> Centers Spine -> Scales.
        """
        if not landmarks:
            return []

        # 1. Update Calibration
        if calibration_data:
            self.spine_center_x = calibration_data.get('spine_center_x', 0.5)
            self.max_reach_y = calibration_data.get('max_reach_y', 0.0)
            self.shoulder_width = calibration_data.get('shoulder_width', 0.5)
            if self.shoulder_width < 0.01: self.shoulder_width = 0.5

        # 2. Detect & Smooth Rotation Angle (Yaw)
        raw_angle = self._get_yaw_angle(landmarks)
        self.smoothed_angle = (self.alpha * raw_angle + (1 - self.alpha) * self.smoothed_angle)

        # 3. Apply Transformations
        aligned = []
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        lsx = self._get_val(l_sh, 'x')
        rsx = self._get_val(r_sh, 'x')
        pivot_x = (lsx + rsx) / 2

        for lm in landmarks:
            x = self._get_val(lm, 'x')
            y = self._get_val(lm, 'y')
            z = self._get_val(lm, 'z')
            vis = self._get_val(lm, 'visibility', 1.0)

            # A. ROTATE
            rot_x, rot_z = self._rotate_point(x, z, pivot_x, -self.smoothed_angle)

            # B. CENTER & SCALE
            norm_x = (rot_x - self.spine_center_x) / self.shoulder_width
            norm_y = (y - self.max_reach_y) / self.shoulder_width

            aligned.append(Landmark(x=norm_x, y=norm_y, z=rot_z, visibility=vis))

        return aligned

    def _get_yaw_angle(self, landmarks):
        l_sh = landmarks[11]; r_sh = landmarks[12]
        lx = self._get_val(l_sh, 'x'); lz = self._get_val(l_sh, 'z')
        rx = self._get_val(r_sh, 'x'); rz = self._get_val(r_sh, 'z')
        return np.arctan2(rz - lz, rx - lx)

    def _rotate_point(self, x, z, pivot_x, angle_rad):
        tx = x - pivot_x
        c = np.cos(angle_rad); s = np.sin(angle_rad)
        rx = tx * c - z * s
        rz = tx * s + z * c
        return rx + pivot_x, rz
