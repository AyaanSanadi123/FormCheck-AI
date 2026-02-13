import numpy as np

class StandingLateralRaiseNormalizer:
    def __init__(self):
        # --- CONFIGURATION ---
        self.alpha = 0.25  # Smoothing factor for rotation delta (EMA)
        self.smoothed_delta = 0
        self.target_angle = 0.0  # We want the user facing perfectly 0° (Frontal)

    def process(self, landmarks):
        """
        Rectifies perspective to a perfect Frontal View centered on shoulders.
        Ensures Y-axis integrity for torso-length based shrug detection.
        """
        if not landmarks or len(landmarks) < 33:
            return []

        # 1. Capture Shoulder Landmarks
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        # 2. Calculate Current Facing Angle (X-Z Plane)
        # Using arctan2 on the Z and X difference between shoulders
        dx = l_sh.x - r_sh.x
        dz = l_sh.z - r_sh.z
        curr_angle = np.degrees(np.arctan2(dz, dx))

        # 3. Determine Correction Delta
        # Goal is to bring curr_angle to 0.0
        delta = self.target_angle - curr_angle

        # 4. Apply EMA Smoothing
        # Stabilizes the 'Virtual Camera' to prevent skeleton jitters
        self.smoothed_delta = (self.alpha * delta) + (1 - self.alpha) * self.smoothed_delta

        # 5. Define THE PIVOT (Mid-Shoulder)
        # This is the 'Anchor' that keeps torso and arm vectors concentric
        pivot_x = (l_sh.x + r_sh.x) / 2
        pivot_z = (l_sh.z + r_sh.z) / 2

        return self._apply_rotation(landmarks, self.smoothed_delta, pivot_x, pivot_z)

    def _apply_rotation(self, landmarks, angle_deg, px, pz):
        """
        Applies X-Z Plane Rectification while strictly locking the Y-axis.
        """
        angle_rad = np.radians(angle_deg)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        normalized_landmarks = []
        for lm in landmarks:
            # Shift coordinate system to origin relative to Mid-Shoulder Pivot
            x_rel = lm.x - px
            z_rel = lm.z - pz

            # Standard 2D rotation matrix applied to the 'Floor' plane
            new_x = x_rel * cos_theta - z_rel * sin_theta
            new_z = x_rel * sin_theta + z_rel * cos_theta

            # Reconstruct the landmark with preserved Y (Gravity)
            normalized_landmarks.append(type(lm)(
                x = new_x + px,
                y = lm.y,  # STATED REQUIREMENT: Strictly lock Y-axis
                z = new_z + pz,
                visibility = lm.visibility
            ))

        return normalized_landmarks