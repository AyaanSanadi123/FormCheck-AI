import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class SquatNormalizer:
    def __init__(self):
        # Default state
        self.active_side = "RIGHT" 
        self.floor_y = 0.5 
        self.scale_factor = 1.0

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Normal Pose (Facing Right, Floor at 0.0, Scaled).
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper (active_side, floor_y, etc.)
        """
        if not landmarks:
            return []

        # 1. Update Calibration (If provided by Gatekeeper)
        if calibration_data:
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.floor_y = calibration_data.get('floor_y', 0.5)
            # Torso length used as scaling factor
            self.scale_factor = calibration_data.get('calibrated_scale', 1.0)

        # 2. Determine Origin (The Anchor)
        # We use the Active Hip as the horizontal origin.
        # But we use the Calibrated Floor as the vertical origin.
        
        # Select indices based on active side
        if self.active_side == "LEFT":
            idx_hip = 23
            idx_ankle = 27
            idx_sh = 11
        else:
            idx_hip = 24
            idx_ankle = 28
            idx_sh = 12

        l_hip = landmarks[idx_hip]
        
        # SAFE EXTRACTION for origin
        origin_x = l_hip.get('x', 0.0) if isinstance(l_hip, dict) else getattr(l_hip, 'x', 0.0)
        
        # Determine Facing Direction
        # Standardize Facing: If active_side is LEFT, user is facing LEFT (typically).
        # We want to flip everything horizontally if Facing Left.
        facing_mult = -1.0 if self.active_side == "LEFT" else 1.0

        aligned = []
        for lm in landmarks:
            # --- SAFE EXTRACTION FIX ---
            if isinstance(lm, dict):
                x = lm.get('x', 0.0)
                y = lm.get('y', 0.0)
                z = lm.get('z', 0.0)
                vis = lm.get('visibility', 1.0)
            else:
                x = getattr(lm, 'x', 0.0)
                y = getattr(lm, 'y', 0.0)
                z = getattr(lm, 'z', 0.0)
                vis = getattr(lm, 'visibility', 1.0)

            # 3. Transform Points
            
            # X: Shift origin to Hip, Flip if facing Left, Scale
            # Result: Hip is at X=0. Knees forward is +X.
            norm_x = ((x - origin_x) * facing_mult) / self.scale_factor

            # Y: Shift origin to Floor, Invert (Up is +), Scale
            # Result: Floor is Y=0. Hip is +Y.
            norm_y = (self.floor_y - y) / self.scale_factor

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned