import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class HamstringCurlNormalizer:
    def __init__(self):
        # Default state (will be updated by Gatekeeper calibration)
        self.facing_side = 1.0  # 1.0 = Right, -1.0 = Left
        self.knee_origin_x = 0.5 
        self.knee_origin_y = 0.5 
        self.leg_length = 1.0   # Scale factor (Tibia/Fibula length)

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Knee-Zeroed, Leg-Scaled grid.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper.
        """
        if not landmarks or len(landmarks) < 29:
            return []

        # 1. Update Calibration (Provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.knee_origin_x = calibration_data.get('knee_origin_x', 0.5)
            self.knee_origin_y = calibration_data.get('knee_origin_y', 0.5)
            self.leg_length = calibration_data.get('leg_length', 1.0)

        def get_attr(lm, attr):
            if isinstance(lm, dict):
                return lm.get(attr)
            return getattr(lm, attr)

        # 2. Determine Local Origin and Scale
        local_origin_x = self.knee_origin_x
        local_origin_y = self.knee_origin_y
        local_scale = self.leg_length

        # Fallback if uncalibrated: calculate dynamically from current frame
        if calibration_data is None and self.knee_origin_x == 0.5:
            l_knee = landmarks[25]
            r_knee = landmarks[26]
            local_origin_x = (get_attr(l_knee, 'x') + get_attr(r_knee, 'x')) / 2
            local_origin_y = (get_attr(l_knee, 'y') + get_attr(r_knee, 'y')) / 2
            
            l_ankle = landmarks[27]
            r_ankle = landmarks[28]
            ank_x = (get_attr(l_ankle, 'x') + get_attr(r_ankle, 'x')) / 2
            ank_y = (get_attr(l_ankle, 'y') + get_attr(r_ankle, 'y')) / 2
            
            local_scale = np.sqrt((ank_x - local_origin_x)**2 + (ank_y - local_origin_y)**2)

        # Prevent division by zero
        if local_scale < 0.001: local_scale = 1.0

        aligned = []
        for lm in landmarks:
            x = get_attr(lm, 'x')
            y = get_attr(lm, 'y')
            z = get_attr(lm, 'z')
            vis = get_attr(lm, 'visibility')
            if vis is None: vis = 1.0

            # 3. Transform Points
            
            # A. SHIFT ORIGIN, FLIP X, and SCALE
            # Result: Knee is (0,0). Feet are in +X direction. Head is in -X direction.
            norm_x = ((x - local_origin_x) * self.facing_side) / local_scale

            # B. SHIFT ORIGIN, INVERT Y, and SCALE
            # Result: Knee is (0,0). Up (towards the ceiling) is +Y.
            norm_y = (local_origin_y - y) / local_scale

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned