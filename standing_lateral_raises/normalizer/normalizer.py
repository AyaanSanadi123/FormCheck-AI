import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class StandingLateralRaiseNormalizer:
    def __init__(self):
        # No state needed for this implementation, as calibration data is passed per frame
        pass

    def process(self, landmarks, calibration_data):
        """
        Main pipeline: 
        1. Get calibration data
        2. Normalize the skeleton (translate, scale, flip/rotate to canonical frontal view)
        """
        if not landmarks:
            return []

        if not calibration_data:
            # Cannot normalize without calibration, return raw
            return landmarks
            
        normalized_landmarks = self._normalize_skeleton(
            landmarks, 
            calibration_data
        )

        return normalized_landmarks

    def _normalize_skeleton(self, landmarks, cal_data):
        """
        Translates, scales, and rotates the skeleton to a canonical frontal pose.
        - Origin at the Mid-Shoulder (11, 12)
        - Scaled by 'scale_factor' (torso_baseline) from calibration_data
        - Rotated to ensure a perfect frontal view (hips/shoulders aligned on X-axis)
        """
        scale_factor = cal_data.get("scale_factor", 1.0)
        
        # Avoid division by zero
        if scale_factor < 1e-5:
            scale_factor = 1.0

        # --- Define Anchor (Mid-Shoulder) ---
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        anchor_x = (l_sh.x + r_sh.x) / 2
        anchor_y = (l_sh.y + r_sh.y) / 2
        anchor_z = (l_sh.z + r_sh.z) / 2

        # --- Calculate current rotation needed to align shoulders/hips on X-axis ---
        # This is the "target_angle = 0.0" equivalent from the original code
        # using the shoulders directly to find the rotation in the X-Z plane
        dx = l_sh.x - r_sh.x
        dz = l_sh.z - r_sh.z
        current_angle = np.degrees(np.arctan2(dz, dx)) # Angle of the line connecting shoulders
        
        # We want this line to be purely horizontal, so the rotation needed is 'current_angle'
        angle_rad = np.radians(current_angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        normalized = []
        for lm in landmarks:
            # 1. Translate to Anchor (Origin)
            x_rel = lm.x - anchor_x
            y_rel = lm.y - anchor_y
            z_rel = lm.z - anchor_z
            
            # 2. Apply Rotation (to make shoulders horizontal)
            # This rotates the whole body so the shoulder line is flat on the X axis
            rotated_x = x_rel * cos_theta - z_rel * sin_theta
            rotated_z = x_rel * sin_theta + z_rel * cos_theta

            # 3. Scale the rotated coordinates
            scaled_x = rotated_x / scale_factor
            scaled_y = y_rel / scale_factor # Y is not rotated, but still scaled
            scaled_z = rotated_z / scale_factor

            # Append the fully normalized landmark
            normalized.append(Landmark(
                x=scaled_x,
                y=scaled_y,
                z=scaled_z,
                visibility=lm.visibility
            ))

        return normalized