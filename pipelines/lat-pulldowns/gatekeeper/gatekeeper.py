import numpy as np
import mediapipe as mp
import time
from collections import deque

class LatPullGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.005 
        
        # The Sliding Window
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # MediaPipe Indices
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        """
        Run per frame.
        Returns: (status: bool, message: str, calibration_data: dict/None)
        """
        if not landmarks:
            self._reset("No user")
            return False, "Looking for lifter...", None

        # --- STEP 1: VISIBILITY CHECK ---
        # We need Upper Body + Hips (to check seating)
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_ELBOW, self.MP_POSE.RIGHT_ELBOW,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx.name} hidden")
                return False, "Ensure Upper Body Visible", None

        # --- STEP 2: ORIENTATION CHECK (BACK VIEW) ---
        # Logic: In a back view, Left Shoulder (11) should be to the LEFT of Right Shoulder (12).
        # (Assuming the camera is not mirrored. If mirrored, this logic flips).
        # Standard webcam feed is usually mirrored. Let's assume standard MediaPipe output.
        
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        # Check: Is Left Shoulder X < Right Shoulder X?
        # Note: If user faces camera (Front), Left(11) is on screen Right (> Right(12)).
        # If user faces away (Back), Left(11) is on screen Left (< Right(12)).
        
        if l_sh.x > r_sh.x:
            self._reset("Wrong Orientation")
            return False, "Turn Around (Back to Camera)", None

        # --- STEP 3: POSE CHECK (ARMS OVERHEAD) ---
        # Wrists should be higher (smaller Y) than Shoulders/Nose
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        nose = landmarks[0]
        
        # Y is inverted (0 is top)
        if l_wrist.y > nose.y or r_wrist.y > nose.y:
            self._reset("Arms too low")
            return False, "Reach Up (Hold the Bar)", None

        # --- STEP 4: STABILITY BUFFER ---
        # We track Wrist Y to ensure they are holding the "Stretch" position
        avg_wrist_y = (l_wrist.y + r_wrist.y) / 2
        
        frame_metrics = {
            "wrist_y": avg_wrist_y,
            "l_sh_x": l_sh.x,
            "r_sh_x": r_sh.x,
            "hip_center_x": (landmarks[23].x + landmarks[24].x) / 2,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Stretch... {progress}%", None

        if not self._is_stable():
            return False, "Stop Swinging...", None

        # --- STEP 5: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks)
        return True, "LAT PULLDOWN READY!", calibration_data

    def _reset(self, reason):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if Wrists have been still."""
        y_history = [m['wrist_y'] for m in self.validation_buffer]
        if np.var(y_history) > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, landmarks):
        """Captures dimensions for Symmetry Analysis."""
        # 1. Max Reach (Zero Point for Rep)
        max_reach_y = np.mean([m['wrist_y'] for m in self.validation_buffer])
        
        # 2. Spine Center (To align the Normalizer)
        # Average of hip center AND shoulder center
        # We need to compute shoulder center from the buffer: (l_sh_x + r_sh_x) / 2
        spine_centers = [
            ((m['l_sh_x'] + m['r_sh_x']) / 2 + m['hip_center_x']) / 2 
            for m in self.validation_buffer
        ]
        spine_center_x = np.mean(spine_centers)
        
        # 3. Shoulder Width (Scale Factor)
        # We average the width over the buffer for robustness
        width_history = [abs(m['l_sh_x'] - m['r_sh_x']) for m in self.validation_buffer]
        shoulder_width = np.mean(width_history)
        
        return {
            "max_reach_y": max_reach_y,
            "spine_center_x": spine_center_x,
            "shoulder_width": shoulder_width,
            "calibrated_at": time.time()
        }