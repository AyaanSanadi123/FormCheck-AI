import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class DeadliftNormalizer:
    def __init__(self):
        # Default state
        self.active_side = "RIGHT"
        self.floor_y = 0.0      # The Y-coordinate of the floor (ankles)
        self.torso_length = 1.0 # Scale factor

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Floor-Zeroed coordinate system.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper (active_side, floor_y, etc.)
        """
        if not landmarks:
            return []

        # 1. Update Calibration (One-time or Continuous)
        if calibration_data:
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.floor_y = calibration_data.get('floor_y', 0.0)
            self.torso_length = calibration_data.get('torso_length', 1.0)

        # 2. Determine Origin (The Anchor)
        # We use the Active Ankle as the horizontal origin.
        if self.active_side == "LEFT":
            idx_ankle = 27
        else:
            idx_ankle = 28
            
        l_ankle = landmarks[idx_ankle]
        origin_x = getattr(l_ankle, 'x', l_ankle.get('x')) if hasattr(l_ankle, 'x') or isinstance(l_ankle, dict) else 0

        # 3. Determine Facing Side (Flip if Left)
        # If user is Left-Facing (Active Side Left), we flip X.
        # This standardizes everyone to face RIGHT (+X).
        facing_mult = -1.0 if self.active_side == "LEFT" else 1.0

        aligned_landmarks = []
        for lm in landmarks:
            x = getattr(lm, 'x', lm.get('x')) if hasattr(lm, 'x') or isinstance(lm, dict) else 0
            y = getattr(lm, 'y', lm.get('y')) if hasattr(lm, 'y') or isinstance(lm, dict) else 0
            z = getattr(lm, 'z', lm.get('z')) if hasattr(lm, 'z') or isinstance(lm, dict) else 0
            vis = getattr(lm, 'visibility', lm.get('visibility', 1.0)) if hasattr(lm, 'visibility') or isinstance(lm, dict) else 1.0

            # 3. Transform Points
            # A. CENTER & FLIP X
            # Shift X to be relative to Ankle (Origin)
            # Then multiply by facing_side.
            norm_x = (x - origin_x) * facing_mult

            # B. ZERO FLOOR & INVERT Y (Solve Gravity)
            # MediaPipe Y is inverted (0=Top, 1=Bottom).
            # We want 0=Floor, +Y=Up.
            norm_y = self.floor_y - y

            # Scale if desired? 
            # Normalizer typically just centers/rotates. Scaling is handled by rep logic using 'torso_length'.
            # But let's keep it 1:1 with screen units for now to match rep logic expectations.

            aligned_landmarks.append(Landmark(
                x=norm_x,
                y=norm_y,
                z=z,
                visibility=vis
            ))

        return aligned_landmarks
