import numpy as np

class Landmark:
    """A simple wrapper to mimic MediaPipe landmark structure."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class SquatNormalizer:
    def __init__(self):
        # Smoothing Factor (alpha)
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
        if self.target_side is None:
            # If user is approx 90 (Right), target 90. If -90 (Left), target -90.
            # Squat gatekeeper ensures they are roughly side-on (60-120 range).
            self.target_side = 90 if raw_angle > 0 else -90

        # 3. Calculate Correction Needed
        # Example: User is at 80°. Target 90°. Correction = +10°.
        raw_correction = self.target_side - raw_angle

        # 4. Smooth the Correction (The "Stabilizer")
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
        Rotates all points around the geometric center (Mid-Hip for Squat)
        by the given angle.
        """
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # Find Anchor (Mid-Hip) to rotate around
        # DYNAMIC: Recalculate per frame to handle shifts
        # Indices: 23=Left Hip, 24=Right Hip
        # Handle both Dict and Object inputs for flexibility
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        l_x = getattr(l_hip, 'x', l_hip.get('x')) if hasattr(l_hip, 'x') or isinstance(l_hip, dict) else 0
        l_z = getattr(l_hip, 'z', l_hip.get('z')) if hasattr(l_hip, 'z') or isinstance(l_hip, dict) else 0
        r_x = getattr(r_hip, 'x', r_hip.get('x')) if hasattr(r_hip, 'x') or isinstance(r_hip, dict) else 0
        r_z = getattr(r_hip, 'z', r_hip.get('z')) if hasattr(r_hip, 'z') or isinstance(r_hip, dict) else 0

        anchor_x = (l_x + r_x) / 2
        anchor_z = (l_z + r_z) / 2

        aligned = []
        for lm in landmarks:
            # Handle input types
            x = getattr(lm, 'x', lm.get('x'))
            y = getattr(lm, 'y', lm.get('y'))
            z = getattr(lm, 'z', lm.get('z'))
            vis = getattr(lm, 'visibility', lm.get('visibility', 1.0))

            # Shift to Anchor (Local Space)
            dx = x - anchor_x
            dz = z - anchor_z
            
            # Apply Rotation Matrix (2D Rotation on X-Z plane)
            # x' = x*cos - z*sin
            # z' = x*sin + z*cos
            new_x = (dx * c) - (dz * s)
            new_z = (dx * s) + (dz * c)

            # Shift back to World Space
            final_x = new_x + anchor_x
            final_z = new_z + anchor_z

            # Return uniform Object structure (Visualizers expect .x .y)
            aligned.append(Landmark(
                x=final_x,
                y=y,      # Y (Gravity) is invariant
                z=final_z,   
                visibility=vis
            ))

        return aligned

    def _get_facing_angle(self, landmarks):
        """Calculates the raw facing angle (-180 to 180) from hips."""
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        l_x = getattr(l_hip, 'x', l_hip.get('x'))
        l_z = getattr(l_hip, 'z', l_hip.get('z'))
        r_x = getattr(r_hip, 'x', r_hip.get('x'))
        r_z = getattr(r_hip, 'z', r_hip.get('z'))

        dx = l_x - r_x
        dz = l_z - r_z
        
        angle_rad = np.arctan2(dz, dx)
        return np.degrees(angle_rad)