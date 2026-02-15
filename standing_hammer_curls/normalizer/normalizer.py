import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class HammerCurlNormalizer:
    def __init__(self):
        # No state needed for this implementation
        pass

    def process(self, landmarks, calibration_data):
        """
        Main pipeline:
        1. Get calibration data
        2. Normalize the skeleton (translate, scale, flip)
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
        - Origin at the active Elbow (13 or 14)
        - Scaled by 'scale_factor' from calibration_data
        - Flipped so the user always faces RIGHT (positive X is forward)
        """
        active_side = cal_data.get("active_side", "RIGHT")
        scale_factor = cal_data.get("scale_factor", 1.0) # Humerus length from Gatekeeper
        
        # Avoid division by zero
        if scale_factor < 1e-5:
            scale_factor = 1.0

        # --- Define Anchor (Active Elbow) ---
        # Assuming right side for simplicity as gatekeeper checks for side profile
        elbow_idx = 14 if active_side == "RIGHT" else 13
        active_elbow = landmarks[elbow_idx]
        
        anchor_x = active_elbow.x
        anchor_y = active_elbow.y
        anchor_z = active_elbow.z

        # --- Determine Flip ---
        # Blueprint: User should always face RIGHT (positive X is forward)
        # In a side profile, if the active side is LEFT, we need to invert the X-axis
        flip_modifier = -1.0 if active_side == "LEFT" else 1.0

        normalized = []
        for lm in landmarks:
            # 1. Translate to Anchor (Origin)
            dx = lm.x - anchor_x
            dy = lm.y - anchor_y
            dz = lm.z - anchor_z
            
            # 2. Scale
            dx_scaled = dx / scale_factor
            dy_scaled = dy / scale_factor
            dz_scaled = dz / scale_factor

            # 3. Flip X and Z for side-view consistency
            dx_flipped = dx_scaled * flip_modifier
            dz_flipped = dz_scaled * flip_modifier


            # Append the fully normalized landmark
            normalized.append(Landmark(
                x=dx_flipped,
                y=dy_scaled,
                z=dz_flipped,
                visibility=lm.visibility
            ))

        return normalized