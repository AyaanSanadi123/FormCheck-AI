import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class ReversePecDeckNormalizer:
    def __init__(self):
        # Smoothing factor for rotation
        self.alpha = 0.2
        self.smoothed_rotation_angle = None

    def process(self, landmarks, calibration_data):
        if not landmarks:
            return []

        scale_factor = calibration_data.get('scale_factor', 1.0)
        
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_hip, r_hip = landmarks[23], landmarks[24]
        
        origin_x = (l_sh.x + r_sh.x) / 2
        origin_y = (l_sh.y + r_sh.y) / 2

        mid_hip_x = (l_hip.x + r_hip.x) / 2
        mid_hip_y = (l_hip.y + r_hip.y) / 2
        
        dx = mid_hip_x - origin_x
        dy = mid_hip_y - origin_y
        raw_angle_to_rotate = -np.arctan2(dx, dy) 

        if self.smoothed_rotation_angle is None:
            self.smoothed_rotation_angle = raw_angle_to_rotate
        else:
            self.smoothed_rotation_angle = (self.alpha * raw_angle_to_rotate + (1 - self.alpha) * self.smoothed_rotation_angle)

        angle_to_rotate = self.smoothed_rotation_angle
        
        normalized_landmarks = []

        for lm in landmarks:
            tx = lm.x - origin_x
            ty = lm.y - origin_y

            rx = tx * np.cos(angle_to_rotate) - ty * np.sin(angle_to_rotate)
            ry = tx * np.sin(angle_to_rotate) + ty * np.cos(angle_to_rotate)

            norm_x = rx / scale_factor
            norm_y = ry / scale_factor
            norm_z = lm.z / scale_factor

            normalized_landmarks.append(Landmark(
                x=norm_x,
                y=norm_y,
                z=norm_z,
                visibility=lm.visibility
            ))

        return normalized_landmarks
