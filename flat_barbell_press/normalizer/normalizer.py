import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class BenchNormalizer:
    def __init__(self):
        # Smoothing Factor (alpha)
        # 0.1 = Heavy smoothing (stable but laggy)
        # 0.9 = Reactive (jittery)
        # 0.2 is a good balance for camera angle correction
        self.alpha = 0.2 
        self.smoothed_correction_angle = None
        
        # State Latching
        self.target_side = None # Will lock to -90 or 90

    def process(self, landmarks):
        """
        Main pipeline: 
        1. Detect Angle
        2. Smooth the Camera Correction
        3. Rotate Landmarks to Perfect Side View
        """
        if not landmarks:
            return []

        # 1. Calculate Raw Facing Angle (Hip Vector)
        raw_angle = self._get_facing_angle(landmarks)

        # 2. Determine Target Side (-90 or +90) - LATCHED
        # We lock this on the first valid frame to prevent flipping 
        # if the user fluctuates near 0/180 due to noise.
        if self.target_side is None:
            self.target_side = -90 if raw_angle < 0 else 90

        # 3. Calculate Correction Needed
        # Example: User is at 75°. Target 90°. Correction = +15°.
        raw_correction = self.target_side - raw_angle

        # 4. Smooth the Correction (The "Stabilizer")
        # We smooth the *correction*, not the landmarks.
        # This stops the world from jittering due to Z-noise.
        if self.smoothed_correction_angle is None:
            self.smoothed_correction_angle = raw_correction
        else:
            self.smoothed_correction_angle = (
                self.alpha * raw_correction + 
                (1 - self.alpha) * self.smoothed_correction_angle
            )

        # 5. Apply Rotation
        # Rotate around the Vertical Y-Axis (Gravity) to fix perspective
        aligned_landmarks = self._rotate_skeleton(
            landmarks, 
            self.smoothed_correction_angle
        )

        return aligned_landmarks

    def _rotate_skeleton(self, landmarks, angle_deg):
        """
        Rotates all points around the geometric center (Mid-Shoulder)
        by the given angle.
        """
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Find Anchor (Mid-Shoulder) to rotate around
        # DYNAMIC: We recalculate this every frame.
        # This ensures that if the user slides up the bench, the pivot moves with them,
        # preventing the "Swinging" artifact.
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        anchor_x = (l_sh.x + r_sh.x) / 2
        anchor_z = (l_sh.z + r_sh.z) / 2

        aligned = []
        for lm in landmarks:
            # Shift to Anchor (Local Space)
            dx = lm.x - anchor_x
            dz = lm.z - anchor_z
            
            # Apply Rotation Matrix (2D Rotation on X-Z plane)
            # x' = x*cos - z*sin
            # z' = x*sin + z*cos
            new_x = (dx * c) - (dz * s)
            new_z = (dx * s) + (dz * c)

            # Shift back to World Space (optional, but keeps visualizer sane)
            final_x = new_x + anchor_x
            final_z = new_z + anchor_z

            # Wrap in object to match MediaPipe interface
            aligned.append(Landmark(
                x=final_x,
                y=lm.y,      # Y (Gravity) is invariant
                z=final_z,   # Depth becomes lateral deviation
                visibility=lm.visibility
            ))

        return aligned

    def _get_facing_angle(self, landmarks):
        """Calculates the angle of the hips relative to the camera."""
        # 23=Left Hip, 24=Right Hip
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        dx = l_hip.x - r_hip.x
        dz = l_hip.z - r_hip.z
        
        # Returns angle in degrees (-180 to 180)
        return np.degrees(np.arctan2(dz, dx))