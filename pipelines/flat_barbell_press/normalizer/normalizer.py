import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class BenchNormalizer:
    def __init__(self):
        # Default state
        self.active_side = "RIGHT"
        self.facing_side = 1.0
        self.shoulder_origin_y = 0.5
        self.arm_length = 1.0

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks: Origin at Active Shoulder, Invert Y, Scale.
        """
        if not landmarks:
            return []

        # 1. Update Calibration
        if calibration_data:
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.shoulder_origin_y = calibration_data.get('bench_y', 0.5)
            self.arm_length = calibration_data.get('arm_length', 1.0)

        # 2. Determine Origin (Active Shoulder)
        if self.active_side == "LEFT":
            idx_sh = 11
        else:
            idx_sh = 12
            
        sh = landmarks[idx_sh]
        
        # --- SAFE EXTRACTION FOR ORIGIN ---
        origin_x = sh.get('x', 0.0) if isinstance(sh, dict) else getattr(sh, 'x', 0.0)
        
        # Scaling factor
        scale = self.arm_length if self.arm_length > 0.01 else 1.0

        aligned = []
        for lm in landmarks:
            # --- SAFE EXTRACTION FIX ---
            if isinstance(lm, dict):
                x = lm.get('x', 0.0)
                y = lm.get('y', 0.0)
                z = lm.get('z', 0.0)
                vis = lm.get('visibility', 1.0)
            else:
                x = getattr(lm, 'x', 0.0)
                y = getattr(lm, 'y', 0.0)
                z = getattr(lm, 'z', 0.0)
                vis = getattr(lm, 'visibility', 1.0)

            # 3. Transform Points
            # X: Shift to shoulder, Flip if facing Left, Scale
            norm_x = ((x - origin_x) * self.facing_side) / scale

            # Y: Shift to Shoulder height, Invert (Up is positive), Scale
            # Note: Bench Y (Shoulder Y) is the baseline. 
            # In MediaPipe Y increases DOWN. So bar ABOVE shoulder has smaller Y.
            # Upward displacement = Shoulder_Y - Bar_Y
            norm_y = (self.shoulder_origin_y - y) / scale

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned