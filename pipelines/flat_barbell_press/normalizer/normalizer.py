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
        origin_x = getattr(sh, 'x', sh.get('x')) if hasattr(sh, 'x') or isinstance(sh, dict) else 0
        
        # Scaling factor
        scale = self.arm_length if self.arm_length > 0.01 else 1.0

        aligned = []
        for lm in landmarks:
            x = getattr(lm, 'x', lm.get('x')) if hasattr(lm, 'x') or isinstance(lm, dict) else 0
            y = getattr(lm, 'y', lm.get('y')) if hasattr(lm, 'y') or isinstance(lm, dict) else 0
            z = getattr(lm, 'z', lm.get('z')) if hasattr(lm, 'z') or isinstance(lm, dict) else 0
            vis = getattr(lm, 'visibility', lm.get('visibility', 1.0))

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
