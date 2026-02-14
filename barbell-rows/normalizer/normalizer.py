import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class BarbellRowNormalizer:
    def __init__(self):
        # Default state (will be updated by Gatekeeper calibration)
        self.facing_side = 1.0  # 1.0 = Right, -1.0 = Left
        self.floor_y = 0.0      # The Y-coordinate of the floor (ankles)
        self.torso_length = 1.0 # Scale factor

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Floor-Zeroed coordinate system.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper (facing_side, floor_y).
        """
        if not landmarks:
            return []

        # 1. Update Calibration (Provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.floor_y = calibration_data.get('floor_y', 0.0)
            self.torso_length = calibration_data.get('torso_length', 1.0)

        # 2. Determine Origin (The Anchor) for X-flipping
        # We use the Mid-Ankle as (0,0) for X.
        def get_attr(lm, attr):
            if isinstance(lm, dict):
                return lm.get(attr)
            return getattr(lm, attr)

        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        
        lx = get_attr(l_ankle, 'x')
        rx = get_attr(r_ankle, 'x')
        ly = get_attr(l_ankle, 'y')
        ry = get_attr(r_ankle, 'y')

        # The X-Origin is the center of the feet
        origin_x = (lx + rx) / 2
        
        # Fallback: If floor_y is 0 (uncalibrated), use current ankle height
        floor_ref = self.floor_y
        if floor_ref == 0.0:
            floor_ref = (ly + ry) / 2

        aligned = []
        for lm in landmarks:
            # Handle input types
            x = get_attr(lm, 'x')
            y = get_attr(lm, 'y')
            z = get_attr(lm, 'z')
            vis = get_attr(lm, 'visibility')
            if vis is None: vis = 1.0

            # 3. Transform Points
            
            # A. FLIP X (Solve Mirroring)
            # Center on ankles, then flip direction if needed.
            norm_x = (x - origin_x) * self.facing_side

            # B. ZERO FLOOR & INVERT Y (Solve Gravity)
            # MediaPipe Y is inverted (0=Top, 1=Bottom).
            # We want 0=Floor, +Y=Up.
            norm_y = floor_ref - y

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned