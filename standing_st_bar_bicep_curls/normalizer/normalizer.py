import numpy as np

class BicepCurlNormalizer:
    def __init__(self):
        # --- CONFIGURATION ---
        self.alpha = 0.25  # Smoothing for the rotation angle
        self.smoothed_angle = 0
        self.target_vertical = -90.0  # Perfect vertical in arctan2 space

    def process(self, landmarks):
        """
        Rectifies the side-profile view. 
        Ensures the Hip-to-Shoulder line is the vertical reference.
        """
        if not landmarks or len(landmarks) < 33:
            return []

        # 1. Capture Profile Pivot (Hip - ID 24) and Shoulder (ID 12)
        # We use the Right side as the primary profile anchor.
        hip = landmarks[24]
        sh = landmarks[12]

        # 2. Calculate Current Spine Lean (Y-X Plane)
        dy = sh.y - hip.y
        dx = sh.x - hip.x
        
        # Angle of the spine relative to the horizontal
        curr_angle = np.degrees(np.arctan2(dy, dx))

        # 3. Determine Correction Delta
        # We want the spine to be at exactly -90 degrees (straight up)
        delta = self.target_vertical - curr_angle

        # 4. Apply EMA Smoothing
        # Prevents "swaying" of the skeleton caused by camera jitter
        self.smoothed_angle = (self.alpha * delta) + (1 - self.alpha) * self.smoothed_angle

        # 5. Apply Rotation around the Hip Pivot
        return self._apply_rotation(landmarks, self.smoothed_angle, hip.x, hip.y)

    def _apply_rotation(self, landmarks, angle_deg, px, py):
        """
        Rotates the skeleton in the Sagittal (X-Y) plane.
        This aligns the user's torso with the gravity axis.
        """
        angle_rad = np.radians(angle_deg)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)

        normalized_landmarks = []
        for lm in landmarks:
            # Shift to Hip Origin
            x_rel = lm.x - px
            y_rel = lm.y - py

            # 2D Rotation (Sagittal Rectification)
            new_x = x_rel * cos_t - y_rel * sin_t
            new_y = x_rel * sin_t + y_rel * cos_t

            normalized_landmarks.append(type(lm)(
                x = new_x + px,
                y = new_y + py,
                z = lm.z, # Z-depth is preserved
                visibility = lm.visibility
            ))

        return normalized_landmarks