import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class OverheadPressNormalizer:
    def __init__(self):
        # No state needed for this implementation
        pass

    def process(self, landmarks, calibration_data):
        """
        Main pipeline: 
        1. Get calibration data
        2. Normalize the skeleton
        """
        if not landmarks:
            return []

        if not calibration_data:
            # Cannot normalize without calibration, return raw
            return landmarks
            
        aligned_landmarks = self._normalize_skeleton(
            landmarks, 
            calibration_data
        )

        return aligned_landmarks

    def _normalize_skeleton(self, landmarks, cal_data):
        """
        Translates, scales, and flips the skeleton to a canonical pose.
        """
        active_side = cal_data.get("active_side", "RIGHT")
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

        # --- Determine Flip ---
        # Blueprint: User should always face RIGHT (positive X is forward)
        # In a side profile, if the active side is LEFT, we need to invert the X-axis
        flip_modifier = -1.0 if active_side == "LEFT" else 1.0

        aligned = []
        for lm in landmarks:
            # 1. Translate to Anchor (Origin)
            dx = lm.x - anchor_x
            dy = lm.y - anchor_y
            dz = lm.z - anchor_z
            
            # 2. Scale
            dx_scaled = dx / scale_factor
            dy_scaled = dy / scale_factor
            dz_scaled = dz / scale_factor

            # 3. Flip
            # We flip the axis perpendicular to the camera for a side view (X)
            # and the depth axis (Z)
            dx_flipped = dx_scaled * flip_modifier
            dz_flipped = dz_scaled * flip_modifier

            # Append the fully normalized landmark
            aligned.append(Landmark(
                x=dx_flipped,
                y=dy_scaled,
                z=dz_flipped,
                visibility=lm.visibility
            ))

        return aligned
