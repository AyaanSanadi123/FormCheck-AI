import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class LungeNormalizer:
    def __init__(self):
        # Smoothing factor for rotation
        self.alpha = 0.2
        self.smoothed_rotation_angle = None

    def process(self, landmarks, calibration_data):
        if not landmarks:
            return []

        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        floor_y_baseline = calibration_data.get('floor_y_baseline', 0.0)
        
        # Determine relevant indices for active side
        ank_idx = 28 if active_side == 'RIGHT' else 27
        toe_idx = 32 if active_side == 'RIGHT' else 31
        
        # 1. Identify Front Ankle (Anchor)
        origin_x, origin_y = landmarks[ank_idx].x, landmarks[ank_idx].y

        # 2. Calculate Floor Rotation
        dx_raw = landmarks[toe_idx].x - landmarks[ank_idx].x
        dy_raw = landmarks[toe_idx].y - landmarks[ank_idx].y
        
        # Angle to make the floor plane perfectly horizontal
        raw_angle_to_rotate = -np.arctan2(dy_raw, dx_raw)

        if self.smoothed_rotation_angle is None:
            self.smoothed_rotation_angle = raw_angle_to_rotate
        else:
            self.smoothed_rotation_angle = (self.alpha * raw_angle_to_rotate + (1 - self.alpha) * self.smoothed_rotation_angle)

        angle_to_rotate = self.smoothed_rotation_angle
        
        normalized_landmarks = []
        for lm in landmarks:
            # A. Translate: Move Front Ankle to (0,0) and adjust for baseline floor
            tx = lm.x - origin_x
            ty = lm.y - origin_y

            # B. Rotation: Standardize floor level
            rx = tx * np.cos(angle_to_rotate) - ty * np.sin(angle_to_rotate)
            ry = tx * np.sin(angle_to_rotate) + ty * np.cos(angle_to_rotate)

            # C. Directional Correction (If user faces left, flip X to keep analysis consistent)
            if active_side == 'LEFT': rx = -rx

            # D. Scale: Standardize by Torso Length
            norm_x = rx / scale_factor
            norm_y = ry / scale_factor
            norm_z = lm.z / scale_factor

            normalized_landmarks.append(Landmark(
                x=norm_x, y=norm_y, z=norm_z, visibility=lm.visibility
            ))

        return normalized_landmarks
