import numpy as np
import mediapipe as mp
import time
from collections import deque

class TricepPushdownGatekeeper:
    def __init__(self):
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5 
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.02 
        
        # Biomechanical Tolerances
        self.MAX_STARTING_ELBOW_ANGLE = 100.0  # Must start with elbows bent
        
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def _get_dominant_side(self, landmarks):
        """
        Determines which side of the body is more visible (closest to camera).
        Returns ("LEFT", indices) or ("RIGHT", indices).
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

        # Check visibility for both sides
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

        # --- STEP 1: DETERMINE DOMINANT SIDE ---
        side_name, indices = self._get_dominant_side(landmarks)
        idx_sh, idx_el, idx_wr, idx_hip = indices

        # --- STEP 2: VISIBILITY CHECK (Dominant Side Only) ---
        for idx in indices:
            if landmarks[idx].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, f"{side_name.title()} Side Not Fully Visible", None

        # Extract coordinates for the dominant side
        sh = landmarks[idx_sh]
        el = landmarks[idx_el]
        wr = landmarks[idx_wr]
        
        # --- STEP 3: DETERMINE FACING DIRECTION ---
        # If wrist x > shoulder x, facing Right (1.0). Else Left (-1.0).
        facing_dir = 1.0 if wr.x > sh.x else -1.0

        # --- STEP 4: UPPER ARM POSTURE CHECK ---
        # Elbows must be below the shoulders (MediaPipe Y increases downwards)
        if el.y < sh.y:
            self._reset()
            return False, "Lower Your Elbows", None

        # --- STEP 5: THE FLEXION CHECK (ELBOWS BENT) ---
        class Point:
            def __init__(self, x, y): self.x, self.y = x, y
            
        elbow_angle = self._calculate_angle(
            Point(sh.x, sh.y),
            Point(el.x, el.y),
            Point(wr.x, wr.y)
        )
        
        if elbow_angle > self.MAX_STARTING_ELBOW_ANGLE:
            self._reset()
            return False, "Bend Elbows (Start at Top)", None

        # --- STEP 6: STABILITY BUFFER ---
        if self.validation_buffer and self.validation_buffer[-1]['side'] != side_name:
            self._reset()

        frame_metrics = {
            "side": side_name,
            "sh_x": sh.x,
            "sh_y": sh.y,
            "el_x": el.x,
            "el_y": el.y,
            "wr_x": wr.x,
            "wr_y": wr.y,
            "elbow_angle": elbow_angle,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Start Position... {progress}%", None

        if not self._is_stable():
            return False, "Hold Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(facing_dir, side_name)
        return True, "PUSHDOWN READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the wrists and elbows have been still."""
        wrist_history = [m['wr_y'] for m in self.validation_buffer]
        elbow_history = [m['el_y'] for m in self.validation_buffer]
        if np.std(wrist_history) > self.STABILITY_VARIANCE or np.std(elbow_history) > self.STABILITY_VARIANCE: 
            return False
        return True

    def _generate_passport(self, facing_dir, side_name):
        # 1. Shoulder Origin (To zero the Normalizer grid)
        shoulder_origin_x = np.mean([m['sh_x'] for m in self.validation_buffer])
        shoulder_origin_y = np.mean([m['sh_y'] for m in self.validation_buffer])
        
        # 2. Capture Baseline Elbow Position (To catch 'Elbow Swing' cheating)
        baseline_el_x = np.mean([m['el_x'] for m in self.validation_buffer])
        baseline_el_y = np.mean([m['el_y'] for m in self.validation_buffer])
        
        # 3. Upper Arm Length (Scale factor)
        upper_arm_length = np.sqrt((baseline_el_x - shoulder_origin_x)**2 + (baseline_el_y - shoulder_origin_y)**2)
        
        return {
            "facing_side": facing_dir,
            "active_side": side_name,
            "shoulder_origin_x": shoulder_origin_x,
            "shoulder_origin_y": shoulder_origin_y,
            "baseline_el_x": baseline_el_x,
            "baseline_el_y": baseline_el_y,
            "arm_length": upper_arm_length,
            "calibrated_at": time.time()
        }

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        return np.degrees(np.arccos(cosine_angle))
