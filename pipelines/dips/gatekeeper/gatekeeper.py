import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    """
    Gatekeeper for Parallel Bar Dips.
    Enforces 75-105 degree azimuth viewing range and captures dynamic perspective scale.
    """
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5 
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.02 
        
        # Biomechanical Tolerances
        self.MIN_STARTING_ANGLE = 160.0 # Arms must be locked out
        self.MAX_AZIMUTH_RATIO = 0.12  # Horizontal shoulder width / Torso length
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def _get_dominant_side(self, landmarks):
        """Determines which side of the body is facing the camera."""
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

        # --- STEP 3: VERTICAL CHAIN CHECK ---
        # Wrist must be the lowest point supporting the body (Y is inverted in MP)
        if not (sh.y < el.y < wr.y):
            self._reset()
            return False, "Mount the bars (Wrists down)", None

        # --- STEP 4: EXTENSION CHECK (Arms Straight) ---
        arm_angle = self._calculate_angle(sh, el, wr)
        if arm_angle < self.MIN_STARTING_ANGLE:
            self._reset()
            return False, "Lock Arms Out to Start", None

        # --- STEP 5: CAMERA AZIMUTH ENFORCEMENT (Shoulder-Width Ratio) ---
        l_sh = landmarks[self.MP_POSE.LEFT_SHOULDER]
        r_sh = landmarks[self.MP_POSE.RIGHT_SHOULDER]
        
        dx_torso = abs(sh.x - hip.x)
        dy_torso = abs(sh.y - hip.y)
        torso_length = np.sqrt(dx_torso**2 + dy_torso**2)
        if torso_length < 0.001: torso_length = 1.0

        # Width between shoulders relative to torso height
        azimuth_ratio = abs(l_sh.x - r_sh.x) / torso_length
        if azimuth_ratio > self.MAX_AZIMUTH_RATIO:
            self._reset()
            return False, "Camera must be directly to your side!", None

        # --- STEP 6: DETERMINE FACING DIRECTION ---
        # Use Nose or Ear (Anterior landmarks) relative to shoulder
        nose = landmarks[self.MP_POSE.NOSE]
        l_ear = landmarks[self.MP_POSE.LEFT_EAR]
        r_ear = landmarks[self.MP_POSE.RIGHT_EAR]
        
        # Priority: Nose -> Ear
        ref_pt = nose if nose.visibility > 0.5 else (l_ear if side_name == "LEFT" else r_ear)
        
        if ref_pt.visibility > 0.4:
            facing_dir = 1.0 if ref_pt.x > sh.x else -1.0
        else:
            # Fallback to hip-shoulder (less reliable but better than nothing)
            facing_dir = 1.0 if sh.x > hip.x else -1.0

        # --- STEP 7: STABILITY BUFFER ---
        if self.validation_buffer and self.validation_buffer[-1]['side'] != side_name:
            self._reset()

        frame_metrics = {
            "side": side_name,
            "sh_x": sh.x, "sh_y": sh.y,
            "hip_x": hip.x, "hip_y": hip.y,
            "el_x": el.x, "el_y": el.y,
            "wr_x": wr.x, "wr_y": wr.y,
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
        return True, "DIPS READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the shoulder and hip have been stable (isometric hold) in both axes."""
        sh_x = [m['sh_x'] for m in self.validation_buffer]
        sh_y = [m['sh_y'] for m in self.validation_buffer]
        hip_x = [m['hip_x'] for m in self.validation_buffer]
        hip_y = [m['hip_y'] for m in self.validation_buffer]
        
        variances = [np.std(sh_x), np.std(sh_y), np.std(hip_x), np.std(hip_y)]
        if any(v > self.STABILITY_VARIANCE for v in variances):
            return False
        return True

    def _generate_passport(self, facing_dir, side_name):
        """Generates the calibration dict. Uses Shoulder as the Primary Mechanical Pivot."""
        avg_sh_x = np.mean([m['sh_x'] for m in self.validation_buffer])
        avg_sh_y = np.mean([m['sh_y'] for m in self.validation_buffer])
        avg_hip_x = np.mean([m['hip_x'] for m in self.validation_buffer])
        avg_hip_y = np.mean([m['hip_y'] for m in self.validation_buffer])
        avg_wr_x = np.mean([m['wr_x'] for m in self.validation_buffer])
        avg_wr_y = np.mean([m['wr_y'] for m in self.validation_buffer])
        
        # Scale factor = Torso Length
        scale_factor = np.sqrt((avg_sh_x - avg_hip_x)**2 + (avg_sh_y - avg_hip_y)**2)
        if scale_factor < 0.001: scale_factor = 1.0
        
        # Calculate signed torso angle (Forward lean vs Backward lean)
        # dy is positive (hip below shoulder), dx is positive if leaning "forward" in facing direction
        dy = avg_hip_y - avg_sh_y
        dx = (avg_sh_x - avg_hip_x) * facing_dir
        setup_torso_angle = np.degrees(np.arctan2(dx, dy))

        return {
            "active_side": side_name,
            "facing_side": facing_dir,
            "shoulder_origin_x": avg_sh_x, # CHANGED: Shoulder is the pivot
            "shoulder_origin_y": avg_sh_y,
            "wrist_baseline_x": avg_wr_x,
            "wrist_baseline_y": avg_wr_y,
            "setup_torso_angle": setup_torso_angle,
            "scale_factor": scale_factor,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        ax, ay = (a.x, a.y) if hasattr(a, 'x') else (a[0], a[1])
        bx, by = (b.x, b.y) if hasattr(b, 'x') else (b[0], b[1])
        cx, cy = (c.x, c.y) if hasattr(c, 'x') else (c[0], c[1])

        ba = np.array([ax - bx, ay - by])
        bc = np.array([cx - bx, cy - by])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0: return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))