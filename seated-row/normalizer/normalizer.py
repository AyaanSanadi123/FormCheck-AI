import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class SeatedRowNormalizer:
    def __init__(self):
        # Default state
        self.active_side = "RIGHT"
        self.scale_factor = 1.0

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing coordinate system.
        """
        if not landmarks:
            return []

        # 1. Update Calibration (One-time or Continuous)
        if calibration_data:
            self.active_side = calibration_data.get('active_side', "RIGHT")
            
        # 2. Determine Origin (Active Hip)
        # We use the Active Hip as the horizontal origin.
        if self.active_side == "LEFT":
            idx_hip = 23
            idx_sh = 11
        else:
            idx_hip = 24
            idx_sh = 12
            
        l_hip = landmarks[idx_hip]
        l_sh = landmarks[idx_sh]
        
        origin_x = getattr(l_hip, 'x', l_hip.get('x')) if hasattr(l_hip, 'x') or isinstance(l_hip, dict) else 0
        origin_y = getattr(l_hip, 'y', l_hip.get('y')) if hasattr(l_hip, 'y') or isinstance(l_hip, dict) else 0

        # Calculate Scale Factor (Torso Length)
        sh_y = getattr(l_sh, 'y', l_sh.get('y')) if hasattr(l_sh, 'y') or isinstance(l_sh, dict) else 0
        scale = abs(sh_y - origin_y)
        if scale < 0.001: scale = 1.0
        self.scale_factor = scale

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
            # Shift X to be relative to Hip (Origin)
            # Then multiply by facing_side.
            norm_x = ((x - origin_x) * facing_mult) / self.scale_factor

            # B. INVERT Y (Solve Gravity)
            # MediaPipe Y is inverted (0=Top, 1=Bottom).
            # We want Hip=0, +Y=Up.
            norm_y = (origin_y - y) / self.scale_factor

            aligned_landmarks.append(Landmark(
                x=norm_x,
                y=norm_y,
                z=z,
                visibility=vis
            ))

        return aligned_landmarks
