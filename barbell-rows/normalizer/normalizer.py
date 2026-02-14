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
        self.active_side = "RIGHT"
        self.floor_y = 0.0      # The Y-coordinate of the floor (ankles)
        self.torso_length = 1.0 # Scale factor

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Floor-Zeroed coordinate system.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper.
        """
        if not landmarks:
            return []

        # 1. Update Calibration (Provided by Gatekeeper)
        if calibration_data:
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.floor_y = calibration_data.get('floor_y', 0.0)
            self.torso_length = calibration_data.get('torso_length', 1.0)

        # 2. Determine Origin (The Anchor) for X-flipping
        # We use the Active Ankle as (0,0) for X.
        def get_attr(lm, attr):
            if isinstance(lm, dict):
                return lm.get(attr)
            return getattr(lm, attr)

        if self.active_side == "LEFT":
            idx_ankle = 27
        else:
            idx_ankle = 28
            
        l_ankle = landmarks[idx_ankle]
        origin_x = get_attr(l_ankle, 'x')
        
        # Fallback: If floor_y is 0 (uncalibrated), use current ankle height
        floor_ref = self.floor_y
        if floor_ref == 0.0:
            floor_ref = get_attr(l_ankle, 'y')

        # 3. Determine Facing Side (Flip if Left)
        # If user is Left-Facing (Active Side Left), we flip X.
        # This standardizes everyone to face RIGHT (+X).
        facing_mult = -1.0 if self.active_side == "LEFT" else 1.0

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
            # Center on active ankle, then flip direction if needed.
            norm_x = (x - origin_x) * facing_mult

            # B. ZERO FLOOR & INVERT Y (Solve Gravity)
            # MediaPipe Y is inverted (0=Top, 1=Bottom).
            # We want 0=Floor, +Y=Up.
            norm_y = floor_ref - y

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned