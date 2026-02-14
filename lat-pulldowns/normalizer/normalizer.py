import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
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

    def _safe_get(self, lm, attr, default=0.0):
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
        # We want the shoulders to be flat (dZ = 0)
        raw_angle = self._get_yaw_angle(landmarks)
        
        # Simple Low-Pass Filter
        self.smoothed_angle = (
            self.alpha * raw_angle + 
            (1 - self.alpha) * self.smoothed_angle
        )

        # 3. Apply Transformations
        aligned = []
        
        # We rotate around the *current* spine center of this frame
        # Indices: 11/12 (Shoulders), 23/24 (Hips) -> Spine Center
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        # Robust access
        lsx = self._safe_get(l_sh, 'x')
        rsx = self._safe_get(r_sh, 'x')
        
        # Pivot point for rotation (X-center of shoulders)
        pivot_x = (lsx + rsx) / 2
        # We don't need pivot_z because Z is relative, we rotate around Z=0 axis effectively

        for lm in landmarks:
            # Handle input types
            x = self._safe_get(lm, 'x')
            y = self._safe_get(lm, 'y')
            z = self._safe_get(lm, 'z')
            vis = self._safe_get(lm, 'visibility', 1.0)

            # A. ROTATE (Yaw Correction)
            # Flatten the view to be parallel to camera
            rot_x, rot_z = self._rotate_point(x, z, pivot_x, -self.smoothed_angle)

            # B. CENTER & SCALE
            # Now we use the ROTATED X coordinates
            norm_x = (rot_x - self.spine_center_x) / self.shoulder_width
            norm_y = (y - self.max_reach_y) / self.shoulder_width

            aligned.append(Landmark(
                x=norm_x,
                y=norm_y,
                z=rot_z,
                visibility=vis
            ))

        return aligned

    def _get_yaw_angle(self, landmarks):
        """Calculates the rotation of the shoulders in the X-Z plane."""
        # 11=Left, 12=Right
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        lx = self._safe_get(l_sh, 'x')
        lz = self._safe_get(l_sh, 'z')
        rx = self._safe_get(r_sh, 'x')
        rz = self._safe_get(r_sh, 'z')

        dx = rx - lx
        dz = rz - lz
        
        # Calculate angle. If dz is 0, angle is 0.
        return np.arctan2(dz, dx)

    def _rotate_point(self, x, z, pivot_x, angle_rad):
        """Rotates a point (x, z) around a pivot (pivot_x, 0)."""
        # Translate to pivot
        tx = x - pivot_x
        tz = z # Z is already relative
        
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        
        # Rotation Matrix Y-axis (Top down view)
        # x' = x*cos - z*sin
        # z' = x*sin + z*cos
        rx = tx * c - tz * s
        rz = tx * s + tz * c
        
        # Translate back
        final_x = rx + pivot_x
        
        return final_x, rz