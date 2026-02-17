import numpy as np

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class TricepPushdownNormalizer:
    def __init__(self):
        # Default state
        self.facing_side = 1.0  
        self.active_side = "RIGHT"
        self.shoulder_origin_x = 0.5 
        self.shoulder_origin_y = 0.5 
        self.arm_length = 0.0 # 0.0 indicates uncalibrated
        self.is_calibrated = False

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Shoulder-Zeroed, Arm-Scaled grid.
        """
        if not landmarks:
            return []

        # 1. Update Calibration (If provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.shoulder_origin_x = calibration_data.get('shoulder_origin_x', 0.5)
            self.shoulder_origin_y = calibration_data.get('shoulder_origin_y', 0.5)
            self.arm_length = calibration_data.get('arm_length', 1.0)
            self.is_calibrated = True

        # 2. Determine Local Origin and Scale from State
        local_origin_x = self.shoulder_origin_x
        local_origin_y = self.shoulder_origin_y
        local_scale = self.arm_length

        # Fallback if uncalibrated: dynamically calculate from the most visible arm
        # This allows the visualizer to work somewhat even before the Gatekeeper passes
        if not self.is_calibrated:
            # Indices for Left (11, 13) and Right (12, 14) Shoulder/Elbow
            l_sh = landmarks[11]
            l_el = landmarks[13]
            r_sh = landmarks[12]
            r_el = landmarks[14]
            
            # Simple visibility check sum
            vis_left = (self._get_val(l_sh, 'visibility') or 0) + (self._get_val(l_el, 'visibility') or 0)
            vis_right = (self._get_val(r_sh, 'visibility') or 0) + (self._get_val(r_el, 'visibility') or 0)
            
            # Pick the more visible side to set temporary scale
            if vis_left > vis_right:
                sh, el = l_sh, l_el
                self.facing_side = -1.0 # Default Left guess
            else:
                sh, el = r_sh, r_el
                self.facing_side = 1.0 # Default Right guess

            local_origin_x = self._get_val(sh, 'x')
            local_origin_y = self._get_val(sh, 'y')
            
            dx = self._get_val(el, 'x') - local_origin_x
            dy = self._get_val(el, 'y') - local_origin_y
            local_scale = np.sqrt(dx*dx + dy*dy)

        # Prevent division by zero
        if local_scale < 0.001: local_scale = 1.0

        aligned = []
        for lm in landmarks:
            x = self._get_val(lm, 'x')
            y = self._get_val(lm, 'y')
            z = self._get_val(lm, 'z')
            vis = self._get_val(lm, 'visibility')
            if vis is None: vis = 1.0

            # 3. Transform Points
            # X: Shift origin to shoulder, flip if facing left, scale by arm length
            # If facing_side is -1 (Left), and x < origin (Left of shoulder), result is positive (Forward).
            norm_x = ((x - local_origin_x) * self.facing_side) / local_scale

            # Y: Shift origin to shoulder, invert Y (UP is positive), scale by arm length
            # MediaPipe Y is 0 at top, 1 at bottom.
            # If shoulder is 0.2, and elbow is 0.5 (below), result: (0.2 - 0.5) = -0.3. Correct.
            norm_y = (local_origin_y - y) / local_scale

            aligned.append(Landmark(x=norm_x, y=norm_y, z=z, visibility=vis))

        return aligned
