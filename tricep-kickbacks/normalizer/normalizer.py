import numpy as np
from pipelines import Landmark

class Normalizer:
    """
    Normalizer for Cable Tricep Kickback.
    Complies with PIPELINE_BLUEPRINT.md standards.
    """
    def __init__(self):
        # Default states before calibration
        self.facing_side = 1.0  
        self.active_side = "RIGHT"
        
        # We use the SHOULDER as the origin for the Kickback
        self.shoulder_origin_x = 0.5 
        self.shoulder_origin_y = 0.5 
        self.shoulder_origin_z = 0.0 
        
        self.scale_factor = 0.0 # 0.0 indicates uncalibrated
        self.is_calibrated = False

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def _normalize_point(self, x, y, z, origin_x, origin_y, origin_z, facing, scale):
        """Thread-safe helper to transform raw points into Canonical Space."""
        if scale < 0.001: return 0.0, 0.0, 0.0
        
        # X: Shift origin to Shoulder, Flip if needed, Scale by Torso Length
        # Result: User faces RIGHT (+X). Kickback moves into -X space.
        norm_x = ((x - origin_x) * facing) / scale
        
        # Y: Shift origin to Shoulder, Invert (Up is +Y), Scale by Torso Length
        norm_y = (origin_y - y) / scale
        
        # Z: Shift origin (relative to shoulder depth), Scale
        norm_z = (z - origin_z) / scale
        
        return norm_x, norm_y, norm_z

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Shoulder-Zeroed, Torso-Scaled grid.
        """
        if not landmarks:
            return []

        # 1. Update Calibration (If provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.shoulder_origin_x = calibration_data.get('shoulder_origin_x', 0.5)
            self.shoulder_origin_y = calibration_data.get('shoulder_origin_y', 0.5)
            self.shoulder_origin_z = calibration_data.get('shoulder_origin_z', 0.0)
            # Gatekeeper now passes Torso Length as scale_factor for consistency
            self.scale_factor = calibration_data.get('scale_factor', 1.0) 
            self.is_calibrated = True

        # 2. Determine Local Origin and Scale for processing
        if self.is_calibrated:
            origin_x, origin_y, origin_z = self.shoulder_origin_x, self.shoulder_origin_y, self.shoulder_origin_z
            facing, scale = self.facing_side, self.scale_factor
        else:
            # Fallback (Uncalibrated Visualizer)
            l_sh = landmarks[11]; l_hip = landmarks[23]
            r_sh = landmarks[12]; r_hip = landmarks[24]
            
            # Visibility fallback
            vis_left = (self._get_val(l_sh, 'visibility') or 0) + (self._get_val(l_hip, 'visibility') or 0)
            vis_right = (self._get_val(r_sh, 'visibility') or 0) + (self._get_val(r_hip, 'visibility') or 0)
            
            if vis_left > vis_right:
                sh, hip = l_sh, l_hip
            else:
                sh, hip = r_sh, r_hip

            origin_x = self._get_val(sh, 'x')
            origin_y = self._get_val(sh, 'y')
            origin_z = self._get_val(sh, 'z')
            
            # Guess facing direction
            facing = 1.0 if origin_x > self._get_val(hip, 'x') else -1.0
            
            # Local scale based on current torso length
            dx = origin_x - self._get_val(hip, 'x')
            dy = origin_y - self._get_val(hip, 'y')
            scale = np.sqrt(dx*dx + dy*dy)
            if scale < 0.001: scale = 1.0

        aligned = []
        for lm in landmarks:
            x = self._get_val(lm, 'x')
            y = self._get_val(lm, 'y')
            z = self._get_val(lm, 'z')
            vis = self._get_val(lm, 'visibility')
            if vis is None: vis = 1.0

            # 3. Transform Points to Canonical Pose via Helper
            nx, ny, nz = self._normalize_point(x, y, z, origin_x, origin_y, origin_z, facing, scale)
            aligned.append(Landmark(x=nx, y=ny, z=nz, visibility=vis))

        return aligned