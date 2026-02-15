import numpy as np
from pipelines import Landmark

class OneArmRowNormalizer:
    def __init__(self):
        # Default states before calibration
        self.facing_side = 1.0  
        self.active_side = "RIGHT"
        
        # We use the HIP as the origin for the One-Arm Row
        self.hip_origin_x = 0.5 
        self.hip_origin_y = 0.5 
        self.hip_origin_z = 0.0 
        
        self.scale_factor = 0.0 # 0.0 indicates uncalibrated
        self.is_calibrated = False

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def _normalize_point(self, x, y, z):
        """Helper to transform any raw point into Canonical Space."""
        if self.scale_factor < 0.001: return 0.0, 0.0, 0.0
        
        # X: Shift origin to Hip, Flip if needed, Scale by Torso Length
        # Result: User always faces RIGHT (+X is forward)
        norm_x = ((x - self.hip_origin_x) * self.facing_side) / self.scale_factor
        
        # Y: Shift origin to Hip, Invert (Up is +Y), Scale by Torso Length
        norm_y = (self.hip_origin_y - y) / self.scale_factor
        
        # Z: Shift origin (relative to hip depth), Scale
        norm_z = (z - self.hip_origin_z) / self.scale_factor
        
        return norm_x, norm_y, norm_z

    def process(self, landmarks, calibration_data=None):
        """
        Standardizes landmarks to a Right-Facing, Hip-Zeroed, Torso-Scaled grid.
        Complies with PIPELINE_BLUEPRINT.md Normalizer requirements.
        """
        if not landmarks:
            return []

        # 1. Update Calibration (If provided by Gatekeeper)
        if calibration_data:
            self.facing_side = calibration_data.get('facing_side', 1.0)
            self.active_side = calibration_data.get('active_side', "RIGHT")
            self.hip_origin_x = calibration_data.get('hip_origin_x', 0.5)
            self.hip_origin_y = calibration_data.get('hip_origin_y', 0.5)
            self.scale_factor = calibration_data.get('scale_factor', 1.0)
            self.is_calibrated = True

        # 2. Determine Local Origin and Scale for Fallback (Uncalibrated Visualizer)
        local_origin_x = self.hip_origin_x
        local_origin_y = self.hip_origin_y
        local_origin_z = self.hip_origin_z
        local_scale = self.scale_factor
        local_facing = self.facing_side

        if not self.is_calibrated:
            l_sh = landmarks[11]; l_hip = landmarks[23]
            r_sh = landmarks[12]; r_hip = landmarks[24]
            
            # Visibility fallback
            vis_left = (self._get_val(l_sh, 'visibility') or 0) + (self._get_val(l_hip, 'visibility') or 0)
            vis_right = (self._get_val(r_sh, 'visibility') or 0) + (self._get_val(r_hip, 'visibility') or 0)
            
            if vis_left > vis_right:
                sh, hip = l_sh, l_hip
            else:
                sh, hip = r_sh, r_hip

            local_origin_x = self._get_val(hip, 'x')
            local_origin_y = self._get_val(hip, 'y')
            local_origin_z = self._get_val(hip, 'z')
            
            # Guess facing direction
            local_facing = 1.0 if self._get_val(sh, 'x') > local_origin_x else -1.0
            
            dx = self._get_val(sh, 'x') - local_origin_x
            dy = self._get_val(sh, 'y') - local_origin_y
            local_scale = np.sqrt(dx*dx + dy*dy)
            
            # Temporarily update instance so helper works this frame
            self.hip_origin_x = local_origin_x
            self.hip_origin_y = local_origin_y
            self.hip_origin_z = local_origin_z
            self.scale_factor = local_scale
            self.facing_side = local_facing

        # Prevent division by zero
        if self.scale_factor < 0.001: self.scale_factor = 1.0

        aligned = []
        for lm in landmarks:
            x = self._get_val(lm, 'x')
            y = self._get_val(lm, 'y')
            z = self._get_val(lm, 'z')
            vis = self._get_val(lm, 'visibility')
            if vis is None: vis = 1.0

            # 3. Transform Points to Canonical Pose via Helper
            nx, ny, nz = self._normalize_point(x, y, z)
            aligned.append(Landmark(x=nx, y=ny, z=nz, visibility=vis))

        return aligned