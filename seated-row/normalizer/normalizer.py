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
        # Default state (will be updated by Gatekeeper calibration)
        self.facing_side = 1.0  # 1.0 = Right, -1.0 = Left
        self.hip_origin_x = 0.5 
        self.hip_origin_y = 0.5 
        self.torso_length = 1.0 # Scale factor for relative distances

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Hip-Zeroed coordinate system.
        Args:
            landmarks: Raw MediaPipe landmarks.
            calibration_data: Dict from Gatekeeper.
        """
        if not landmarks or len(landmarks) < 25:
            return []

        # 1. Update Calibration (Provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.hip_origin_x = calibration_data.get('hip_origin_x', 0.5)
            self.hip_origin_y = calibration_data.get('hip_origin_y', 0.5)
            self.torso_length = calibration_data.get('torso_length', 1.0)

        def get_attr(lm, attr):
            if isinstance(lm, dict):
                return lm.get(attr)
            return getattr(lm, attr)

        # 2. Determine Local Origin and Scale
        # Use calibrated values, but fallback to current frame if never calibrated
        local_origin_x = self.hip_origin_x
        local_origin_y = self.hip_origin_y
        local_scale = self.torso_length

        if calibration_data is None and self.hip_origin_x == 0.5:
            l_hip = landmarks[23]
            r_hip = landmarks[24]
            local_origin_x = (get_attr(l_hip, 'x') + get_attr(r_hip, 'x')) / 2
            local_origin_y = (get_attr(l_hip, 'y') + get_attr(r_hip, 'y')) / 2
            
            # Estimate torso length for scaling if uncalibrated
            l_sh = landmarks[11]
            r_sh = landmarks[12]
            sh_x = (get_attr(l_sh, 'x') + get_attr(r_sh, 'x')) / 2
            sh_y = (get_attr(l_sh, 'y') + get_attr(r_sh, 'y')) / 2
            local_scale = np.sqrt((sh_x - local_origin_x)**2 + (sh_y - local_origin_y)**2)

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
            # Result: Hip is (0,0). Reaching forward is +X, pulling is -X (towards 0).
            norm_x = ((x - local_origin_x) * self.facing_side) / local_scale

            # B. SHIFT ORIGIN, INVERT Y, and SCALE
            # Result: Hip is (0,0). Up is +Y.
            norm_y = (local_origin_y - y) / local_scale

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned