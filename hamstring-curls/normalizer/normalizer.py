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
        self.active_side = "RIGHT" 
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
            self.active_side = calibration_data.get('active_side', "RIGHT")
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

        # Fallback if uncalibrated: calculate dynamically from active knee
        # (This is rare since Gatekeeper should pass data, but good for robustness)
        if calibration_data is None and self.knee_origin_x == 0.5:
            # Simple heuristic fallback
            if self.active_side == "LEFT":
                knee = landmarks[25]
                ankle = landmarks[27]
            else:
                knee = landmarks[26]
                ankle = landmarks[28]
            
            local_origin_x = get_attr(knee, 'x')
            local_origin_y = get_attr(knee, 'y')
            
            ank_x = get_attr(ankle, 'x')
            ank_y = get_attr(ankle, 'y')
            local_scale = np.sqrt((ank_x - local_origin_x)**2 + (ank_y - local_origin_y)**2)

        # Prevent division by zero
        if local_scale < 0.001: local_scale = 1.0

        # 3. Determine Facing Side (Flip if Left)
        # If user is Left-Facing (Active Side Left), we flip X.
        # This standardizes everyone to face RIGHT (+X).
        facing_mult = -1.0 if self.active_side == "LEFT" else 1.0

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
            norm_x = ((x - local_origin_x) * facing_mult) / local_scale

            # B. SHIFT ORIGIN, INVERT Y, and SCALE
            # Result: Knee is (0,0). Up (towards the ceiling) is +Y.
            norm_y = (local_origin_y - y) / local_scale

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned