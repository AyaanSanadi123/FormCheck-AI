import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class HangingLegRaisesNormalizer:
    def __init__(self):
        # Smoothing factor for rotation
        self.alpha = 0.2
        self.smoothed_rotation_angle = None

    def process(self, landmarks, calibration_data):
        if not landmarks:
            return []

        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        bar_origin_x, bar_origin_y = calibration_data.get('bar_origin', (0,0))
        
        # 1. Identify Anchor (The Bar/Wrist)
        sh_idx = 12 if active_side == 'RIGHT' else 11
        wr_idx = 16 if active_side == 'RIGHT' else 15
        
        # Calculate raw angle from current wrist and shoulder
        dx_raw = landmarks[sh_idx].x - landmarks[wr_idx].x
        dy_raw = landmarks[sh_idx].y - landmarks[wr_idx].y
        
        # Angle to make the hang perfectly vertical
        raw_angle_to_rotate = -np.arctan2(dx_raw, dy_raw) 

        if self.smoothed_rotation_angle is None:
            self.smoothed_rotation_angle = raw_angle_to_rotate
        else:
            self.smoothed_rotation_angle = (self.alpha * raw_angle_to_rotate + (1 - self.alpha) * self.smoothed_rotation_angle)

        angle_to_rotate = self.smoothed_rotation_angle
        
        normalized_landmarks = []
        for lm in landmarks:
            # A. Translate: Move Bar Origin to (0,0)
            tx, ty = lm.x - bar_origin_x, lm.y - bar_origin_y

            # B. Rotation: Standardize Verticality
            rx = tx * np.cos(angle_to_rotate) - ty * np.sin(angle_to_rotate)
            ry = tx * np.sin(angle_to_rotate) + ty * np.cos(angle_to_rotate)

            # C. Directional Correction (Standardize facing direction)
            # If user faces left, flip X to keep legs moving into +X space.
            if active_side == 'LEFT': rx = -rx

            # D. Scale: Standardize by Torso Length
            norm_x = rx / scale_factor
            norm_y = ry / scale_factor
            norm_z = lm.z / scale_factor

            normalized_landmarks.append(Landmark(
                x=norm_x, y=norm_y, z=norm_z, visibility=lm.visibility
            ))

        return normalized_landmarks
