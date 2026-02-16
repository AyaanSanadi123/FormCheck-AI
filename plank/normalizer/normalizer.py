import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class PlankNormalizer:
    def __init__(self):
        # Smoothing factor for rotation
        self.alpha = 0.2
        self.smoothed_rotation_angle = None

    def process(self, landmarks, calibration_data):
        if not landmarks:
            return []

        active_side = calibration_data.get('active_side', 'RIGHT')
        scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # Determine relevant indices for active side
        sh_idx = 12 if active_side == 'RIGHT' else 11
        el_idx = 14 if active_side == 'RIGHT' else 13
        ank_idx = 28 if active_side == 'RIGHT' else 27
        
        # 1. Identify Anchor (Elbow)
        origin_x, origin_y = landmarks[el_idx].x, landmarks[el_idx].y

        # 2. Calculate "Structural Rotation"
        dx_raw = landmarks[ank_idx].x - landmarks[sh_idx].x
        dy_raw = landmarks[ank_idx].y - landmarks[sh_idx].y
        
        # Angle to make the Shoulder-Ankle vector horizontal
        raw_angle_to_rotate = -np.arctan2(dy_raw, dx_raw)

        if self.smoothed_rotation_angle is None:
            self.smoothed_rotation_angle = raw_angle_to_rotate
        else:
            self.smoothed_rotation_angle = (self.alpha * raw_angle_to_rotate + (1 - self.alpha) * self.smoothed_rotation_angle)

        angle_to_rotate = self.smoothed_rotation_angle
        
        normalized_landmarks = []
        for lm in landmarks:
            # A. Translation: Move Elbow to (0,0)
            tx, ty = lm.x - origin_x, lm.y - origin_y

            # B. Rotation: Align body beam to horizontal
            rx = tx * np.cos(angle_to_rotate) - ty * np.sin(angle_to_rotate)
            ry = tx * np.sin(angle_to_rotate) + ty * np.cos(angle_to_rotate)

            # C. Facing & Scale
            if active_side == 'LEFT': rx = -rx
            
            norm_x = rx / scale_factor
            norm_y = ry / scale_factor
            norm_z = lm.z / scale_factor

            normalized_landmarks.append(Landmark(
                x=norm_x, y=norm_y, z=norm_z, visibility=lm.visibility
            ))

        return normalized_landmarks
