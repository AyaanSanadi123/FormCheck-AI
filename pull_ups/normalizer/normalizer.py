import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class PullUpsNormalizer:
    def __init__(self):
        # Smoothing factor for rotation
        self.alpha = 0.2
        self.smoothed_rotation_angle = None

    def process(self, landmarks, calibration_data):
        if not landmarks:
            return []

        scale_factor = calibration_data.get('scale_factor', 1.0)
        bar_y_baseline = calibration_data.get('bar_y_baseline', 0.0)
        
        # 1. Identify Mid-Shoulder as the vertical tracking point
        mid_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        
        # 2. Calculate Rotation Angle (Ensure shoulders are level)
        dx_raw = landmarks[12].x - landmarks[11].x
        dy_raw = landmarks[12].y - landmarks[11].y
        
        raw_angle_to_rotate = -np.arctan2(dy_raw, dx_raw)

        if self.smoothed_rotation_angle is None:
            self.smoothed_rotation_angle = raw_angle_to_rotate
        else:
            self.smoothed_rotation_angle = (self.alpha * raw_angle_to_rotate + (1 - self.alpha) * self.smoothed_rotation_angle)

        angle_to_rotate = self.smoothed_rotation_angle
        
        normalized_landmarks = []
        for lm in landmarks:
            # A. Translate to Bar-Y and Torso-X
            tx = lm.x - mid_sh_x
            ty = lm.y - bar_y_baseline

            # B. Rotate for Camera Tilt
            rx = tx * np.cos(angle_to_rotate) - ty * np.sin(angle_to_rotate)
            ry = tx * np.sin(angle_to_rotate) + ty * np.cos(angle_to_rotate)

            # C. Pendulum Correction (Adjust Y based on X-drift)
            # This helps to factor out swinging from vertical progress
            hip_x_normalized = rx # Assuming hip X is correlated with general body X
            # Simplified compensation: penalize vertical lift if there's significant horizontal movement
            # A more advanced model might use a more complex inverse kinematic correction
            # For now, let's keep it simple: no direct Y compensation, just X-drift for faults
            
            # D. Scale to Torso Units
            norm_x = rx / scale_factor
            norm_y = ry / scale_factor
            norm_z = lm.z / scale_factor

            normalized_landmarks.append(Landmark(
                x=norm_x, y=norm_y, z=norm_z, visibility=lm.visibility
            ))

        return normalized_landmarks
