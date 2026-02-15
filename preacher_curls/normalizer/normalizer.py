import numpy as np

class Normalizer:
    """
    Standardizes landmarks for Preacher Curl analysis.
    1. Origin: Elbow (Primary Pivot) to (0,0).
    2. Facing: Ensures user always faces RIGHT (forearm moves in +X).
    3. Scale: Normalizes by scale_factor (Humerus Length).
    """

    def process(self, landmarks, calibration_data):
        """
        Transforms raw landmarks into a standardized coordinate system.
        """
        if not landmarks or not calibration_data:
            return []

        # 1. Extract Calibration Baselines
        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # 2. Identify Primary Pivot: The Elbow (ID 14 for Right, 13 for Left)
        pivot_idx = 14 if active_side == 'RIGHT' else 13
        pivot_raw = landmarks[pivot_idx]
        origin_x, origin_y = pivot_raw.x, pivot_raw.y

        normalized_landmarks = []

        for lm in landmarks:
            # A. Translation: Move Elbow to (0,0)
            temp_x = lm.x - origin_x
            temp_y = lm.y - origin_y

            # B. Facing: Reflection if facing left
            # We want the curl to travel towards the right side of the screen
            if active_side == 'LEFT':
                temp_x = -temp_x

            # C. Scale: Standardize by Humerus Length
            # This makes the upper arm length exactly 1.0 in coordinate space
            norm_x = temp_x / scale_factor
            norm_y = temp_y / scale_factor
            norm_z = lm.z / scale_factor

            # D. Re-encapsulate into landmark objects
            normalized_landmarks.append(type(lm)(
                x=norm_x,
                y=norm_y,
                z=norm_z,
                visibility=lm.visibility
            ))

        return normalized_landmarks