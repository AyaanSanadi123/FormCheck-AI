import numpy as np
import mediapipe as mp
import time
from collections import deque

class BenchGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65 # Slightly lower for bench as arms move fast
        self.STABILITY_VARIANCE = 0.005 # Wrist stability (Bar must be still)
        self.HORIZONTAL_TOLERANCE = 0.20 # Max Y-diff between Shoulder & Hip
        self.ANGLE_TOLERANCE = (60, 120) # Side View Only
        
        # The Sliding Window
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.is_calibrated = False
        
        # MediaPipe Indices
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks):
        """
        Run per frame.
        Returns: (status: bool, message: str, calibration_data: dict/None)
        """
        if not landmarks:
            self._reset("No user detected")
            return False, "Looking for lifter...", None

        # --- STEP 1: VISIBILITY CHECK ---
        # Critical for Bench: Wrists(15,16), Elbows(13,14), Shoulders(11,12)
        # We also check Hips(23,24) to detect bridging
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_ELBOW, self.MP_POSE.RIGHT_ELBOW,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx.name} hidden")
                return False, "Ensure Arms & Hips are visible", None

        # --- STEP 2: HORIZONTAL ORIENTATION CHECK ---
        # User must be lying down. Compare Shoulder Y vs Hip Y.
        # Note: In MediaPipe, Y increases downwards.
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        if abs(shoulder_y - hip_y) > self.HORIZONTAL_TOLERANCE:
            self._reset("User not flat")
            return False, "Lie flat on the bench", None

        # --- STEP 3: ANGLE CHECK ---
        # We calculate the angle LOCALLY (no external import)
        angle = self._calculate_facing_angle(landmarks)
        
        # Strict Range: 60 to 120 degrees (Side View)
        # We take abs() to handle both Left-facing (-90) and Right-facing (+90)
        if not (self.ANGLE_TOLERANCE[0] < abs(angle) < self.ANGLE_TOLERANCE[1]):
            self._reset(f"Bad Angle: {int(angle)}")
            return False, "Camera must be Side-On (60-120 deg)", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        # We track the WRIST position (The Bar) for stability
        bar_y = (landmarks[15].y + landmarks[16].y) / 2
        
        frame_metrics = {
            "bar_y": bar_y,
            "shoulder_y": shoulder_y,
            "angle": angle,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Bar Steady... {progress}%", None

        if not self._is_stable():
            return False, "Don't move the bar...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks)
        return True, "BENCH PRESS READY!", calibration_data

    def _reset(self, reason):
        """Clears the buffer. User has to start the 2s timer over."""
        self.validation_buffer.clear()

    def _calculate_facing_angle(self, landmarks):
        """
        Calculates the angle of the hips relative to the camera.
        0 = Head on, 90 = Side view.
        """
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        
        dx = l_hip.x - r_hip.x
        dz = l_hip.z - r_hip.z
        
        angle_rad = np.arctan2(dz, dx)
        return np.degrees(angle_rad)

    def _is_stable(self):
        """Checks if the BAR (Wrist Y) has been still."""
        bar_history = [m['bar_y'] for m in self.validation_buffer]
        
        # Calculate Variance
        var_bar = np.var(bar_history)
        
        if var_bar > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, landmarks):
        """Captures the user's dimensions for the Rep Logic."""
        avg_bench_height = np.mean([m['shoulder_y'] for m in self.validation_buffer])
        avg_angle = np.mean([m['angle'] for m in self.validation_buffer])
        
        # Calculate Arm Length (Shoulder to Wrist 3D distance)
        # This acts as our "Scale" to determine if they hit chest depth
        s = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        w = np.array([landmarks[15].x, landmarks[15].y, landmarks[15].z])
        arm_length = np.linalg.norm(s - w)
        
        return {
            "bench_y": avg_bench_height,      # The "Zero" line for bridging
            "calibrated_angle": avg_angle,    # For Normalizer
            "arm_length": arm_length,         # For Depth Check
            "calibrated_at": time.time()
        }