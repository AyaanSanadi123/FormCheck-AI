import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    """
    Gatekeeper for Incline Barbell/Dumbbell Press.
    Complies with PIPELINE_BLUEPRINT.md standards.
    """
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5 
        # Stability buffer to ensure the user is holding still
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.02 
        
        # Biomechanical Tolerances
        self.MIN_TORSO_ANGLE = 25.0    # Must be on an incline (not flat)
        self.MAX_TORSO_ANGLE = 65.0    # Must not be completely vertical
        self.MIN_STARTING_ANGLE = 150.0 # Arms must be extended upward (lockout)
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def _get_dominant_side(self, landmarks):
        """
        Determines which side of the body is facing the camera.
        """
        left_indices = [
            self.MP_POSE.LEFT_SHOULDER.value,
            self.MP_POSE.LEFT_ELBOW.value,
            self.MP_POSE.LEFT_WRIST.value,
            self.MP_POSE.LEFT_HIP.value
        ]
        right_indices = [
            self.MP_POSE.RIGHT_SHOULDER.value,
            self.MP_POSE.RIGHT_ELBOW.value,
            self.MP_POSE.RIGHT_WRIST.value,
            self.MP_POSE.RIGHT_HIP.value
        ]

        left_vis = sum(landmarks[i].visibility for i in left_indices)
        right_vis = sum(landmarks[i].visibility for i in right_indices)

        if left_vis > right_vis:
            return "LEFT", left_indices
        else:
            return "RIGHT", right_indices

    def check(self, landmarks):
        """
        Main check method returning (passed, message, calibration_data).
        """
        if not landmarks:
            self._reset()
            return False, "Looking for lifter...", None

        # --- STEP 1: DETERMINE ACTIVE SIDE ---
        side_name, indices = self._get_dominant_side(landmarks)
        idx_sh, idx_el, idx_wr, idx_hip = indices

        # --- STEP 2: VISIBILITY CHECK ---
        for idx in indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, f"{side_name.title()} Side Not Fully Visible", None

        sh = landmarks[idx_sh]
        el = landmarks[idx_el]
        wr = landmarks[idx_wr]
        hip = landmarks[idx_hip]

        # --- STEP 3: THE INCLINE CHECK ---
        dx_torso = abs(sh.x - hip.x)
        dy_torso = abs(sh.y - hip.y)
        if dx_torso < 0.001: dx_torso = 0.001
        
        torso_angle = np.degrees(np.arctan2(dy_torso, dx_torso))
        if torso_angle < self.MIN_TORSO_ANGLE:
            self._reset()
            return False, "Bench is too flat (Raise Incline)", None
        if torso_angle > self.MAX_TORSO_ANGLE:
            self._reset()
            return False, "Bench is too high (Lower Incline)", None

        # --- STEP 4: DETERMINE FACING DIRECTION ---
        # If shoulder X < hip X, the head is on the Left. 
        facing_dir = 1.0 if sh.x < hip.x else -1.0

        # --- STEP 5: VERTICAL ARM CHECK ---
        # Wrist must be above Elbow, and Elbow must be above Shoulder (Y decreases UP)
        if not (wr.y < el.y < sh.y):
            self._reset()
            return False, "Press Weight Upwards to Start", None

        # --- STEP 6: EXTENSION CHECK (Arms Straight) ---
        arm_angle = self._calculate_angle(sh, el, wr)
        
        if arm_angle < self.MIN_STARTING_ANGLE:
            self._reset()
            return False, "Lock Arms Out to Start", None

        # --- STEP 7: STABILITY BUFFER ---
        if self.validation_buffer and self.validation_buffer[-1]['side'] != side_name:
            self._reset()

        frame_metrics = {
            "side": side_name,
            "sh_x": sh.x, "sh_y": sh.y,
            "hip_x": hip.x, "hip_y": hip.y,
            "el_x": el.x, "el_y": el.y,
            "wr_x": wr.x, "wr_y": wr.y,
            "torso_angle": torso_angle,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Start Position... {progress}%", None

        if not self._is_stable():
            return False, "Hold Still...", None

        # --- STEP 8: CALIBRATION PASSPORT ---
        calibration_data = self._generate_passport(facing_dir, side_name)
        return True, "INCLINE PRESS READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the wrist and elbow have been stable (isometric hold) in both axes."""
        wrist_x = [m['wr_x'] for m in self.validation_buffer]
        wrist_y = [m['wr_y'] for m in self.validation_buffer]
        elbow_x = [m['el_x'] for m in self.validation_buffer]
        elbow_y = [m['el_y'] for m in self.validation_buffer]
        
        # Check standard deviation across both X and Y
        variances = [np.std(wrist_x), np.std(wrist_y), np.std(elbow_x), np.std(elbow_y)]
        if any(v > self.STABILITY_VARIANCE for v in variances):
            return False
        return True

    def _generate_passport(self, facing_dir, side_name):
        """
        Generates the calibration dict containing active_side and scale_factor.
        """
        shoulder_origin_x = np.mean([m['sh_x'] for m in self.validation_buffer])
        shoulder_origin_y = np.mean([m['sh_y'] for m in self.validation_buffer])
        
        hip_x = np.mean([m['hip_x'] for m in self.validation_buffer])
        hip_y = np.mean([m['hip_y'] for m in self.validation_buffer])
        
        baseline_el_x = np.mean([m['el_x'] for m in self.validation_buffer])
        baseline_el_y = np.mean([m['el_y'] for m in self.validation_buffer])
        
        setup_torso_angle = np.mean([m['torso_angle'] for m in self.validation_buffer])
        
        # Standardized scale_factor per the Pipeline Blueprint (Torso Length)
        scale_factor = np.sqrt((shoulder_origin_x - hip_x)**2 + (shoulder_origin_y - hip_y)**2)
        
        return {
            "active_side": side_name,
            "facing_side": facing_dir,
            "shoulder_origin_x": shoulder_origin_x,
            "shoulder_origin_y": shoulder_origin_y,
            "baseline_el_x": baseline_el_x,
            "baseline_el_y": baseline_el_y,
            "setup_torso_angle": setup_torso_angle,
            "scale_factor": scale_factor,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        """Calculates angle between three points (a, b, c) where b is the vertex."""
        # Check if points are landmarks (with .x, .y) or simple objects
        ax, ay = (a.x, a.y) if hasattr(a, 'x') else (a[0], a[1])
        bx, by = (b.x, b.y) if hasattr(b, 'x') else (b[0], b[1])
        cx, cy = (c.x, c.y) if hasattr(c, 'x') else (c[0], c[1])

        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))