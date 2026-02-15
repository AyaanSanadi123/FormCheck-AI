import numpy as np

class Normalizer:
    def __init__(self):
        """
        Initializes the Normalizer for Seated Leg Extensions.
        Standardizes orientation, scale, and origin based on Blueprint Section 2.B.
        """
        pass

    def process(self, landmarks, calibration_data):
        """
        Standardizes raw landmarks into a Canonical Pose.
        
        Requirements:
        1. Origin: Translate Knee (Primary Pivot) to (0,0).
        2. Facing: Flip X so user always faces RIGHT.
        3. Scale: Normalize by scale_factor (Thigh Length).
        """
        if not landmarks or not calibration_data:
            return []

        # 1. Setup Baselines from Calibration
        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # Primary Pivot: Knee (26 for Right side, 25 for Left side)
        pivot_idx = 26 if active_side == 'RIGHT' else 25
        pivot_raw = landmarks[pivot_idx]
        origin_x, origin_y = pivot_raw.x, pivot_raw.y

        normalized_landmarks = []

        for lm in landmarks:
            # A. Translation: Move Pivot to (0,0)
            temp_x = lm.x - origin_x
            temp_y = lm.y - origin_y

            # B. Facing: Ensure user faces RIGHT (Positive X is forward)
            # If the active side is LEFT, we flip the X-axis.
            if active_side == 'LEFT':
                temp_x = -temp_x

            # C. Scale: Divide by scale_factor
            # This makes the thigh length exactly 1.0 in coordinate space.
            norm_x = temp_x / scale_factor
            norm_y = temp_y / scale_factor
            norm_z = lm.z / scale_factor # Scale depth consistently

            # Create new landmark object preserving visibility
            normalized_landmarks.append(type(lm)(
                x=norm_x,
                y=norm_y,
                z=norm_z,
                visibility=lm.visibility
            ))

        return normalized_landmarks