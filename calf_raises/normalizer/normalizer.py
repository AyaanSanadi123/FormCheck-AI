import numpy as np

class Normalizer:
    """
    Standardizes raw landmarks for Calf Raises by translating the toe to origin,
    scaling by torso length, and ensuring a consistent right-facing orientation.
    """

    def process(self, landmarks, calibration_data):
        """
        Transforms raw landmarks into a standardized coordinate system.
        
        Requirements:
        1. Origin: Translate the Toe (Primary Pivot) to (0,0).
        2. Facing: Flip X-coordinates if the user is facing LEFT so they always face RIGHT.
        3. Scale: Normalize all coordinates by the scale_factor (Torso Length).
        """
        if not landmarks or not calibration_data:
            return []

        # 1. Extract Calibration Baselines
        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # 2. Identify Primary Pivot: The Toe (ID 32 for Right, 31 for Left)
        pivot_idx = 32 if active_side == 'RIGHT' else 31
        pivot_raw = landmarks[pivot_idx]
        origin_x, origin_y = pivot_raw.x, pivot_raw.y

        normalized_landmarks = []

        for lm in landmarks:
            # A. Translation: Move Toe to (0,0)
            temp_x = lm.x - origin_x
            temp_y = lm.y - origin_y

            # B. Facing: Ensure user always faces RIGHT (Positive X is forward)
            if active_side == 'LEFT':
                temp_x = -temp_x

            # C. Scale: Divide by scale_factor
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