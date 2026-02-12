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
        # Default state (will be updated by Gatekeeper calibration)
        self.facing_side = 1.0  # 1.0 = Right, -1.0 = Left
        self.floor_y = 0.0      # The Y-coordinate of the floor (ankles)
        self.torso_length = 1.0 # Scale factor

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Floor-Zeroed coordinate system.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper (facing_side, floor_y, etc.)
        """
        if not landmarks:
            return []

        # 1. Update Calibration (One-time or Continuous)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.floor_y = calibration_data.get('floor_y', 0.0)
            # We can also get torso_length if needed for scaling X-axis later

        aligned_landmarks = []
        
        # 2. Determine Origin (The Anchor)
        # We use the Mid-Ankle as (0,0) for X and Y in the local space
        # This keeps numbers small and centered on the lift.
        l_ankle = landmarks[27]
        r_ankle = landmarks[28]
        origin_x = (l_ankle.x + r_ankle.x) / 2
        # origin_y is self.floor_y (provided by Gatekeeper)

        # 3. Transform Points
        for lm in landmarks:
            # A. CENTER & FLIP X (Solve Mirroring)
            # Shift X to be relative to Ankle (Origin)
            # Then multiply by facing_side.
            # If Facing Right (+1): (x - origin) * 1 -> Positive is Forward.
            # If Facing Left (-1): (x - origin) * -1 -> Positive is Forward (flipped).
            
            norm_x = (lm.x - origin_x) * self.facing_side

            # B. ZERO FLOOR & INVERT Y (Solve Gravity)
            # MediaPipe Y is inverted (0=Top, 1=Bottom).
            # We want 0=Floor, +Y=Up.
            # Logic: Distance from floor = Floor_Y - Landmark_Y
            # Example: Floor=0.9, Knee=0.7. Height = 0.2.
            
            norm_y = self.floor_y - lm.y

            aligned_landmarks.append(Landmark(
                x=norm_x,
                y=norm_y,
                z=lm.z,
                visibility=lm.visibility
            ))

        return aligned_landmarks