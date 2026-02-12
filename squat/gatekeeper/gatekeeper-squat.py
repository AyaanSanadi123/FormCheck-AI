import numpy as np
import mediapipe as mp
import time
from collections import deque

class Gatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 2.0  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 60 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.85
        self.STABILITY_VARIANCE = 0.015  # Strict stillness
        self.ANGLE_STABILITY_THRESH = 5.0 # Max angle deviation
        self.ANGLE_TOLERANCE = (60, 120) # Accept Side/Oblique, Reject Front
        
        # The Sliding Window (Stores metrics, not full frames)
        self.validation_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # MediaPipe Indices
        self.MP_POSE = mp.solutions.pose.PoseLandmark
        self.CRITICAL_POINTS = [
            self.MP_POSE.LEFT_ANKLE, self.MP_POSE.RIGHT_ANKLE, 
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.NOSE
        ]

    def check(self, landmarks):
        """
        Run per frame.
        Returns: (status: bool, message: str, calibration_data: dict/None)
        """
        if not landmarks:
            self._reset("No user detected")
            return False, "Looking for a human...", None

        # --- STEP 1: PRE-CALCULATIONS & HIP CHECK ---
        # We need hips visible to calculate angle and torso.
        left_hip = landmarks[self.MP_POSE.LEFT_HIP.value]
        right_hip = landmarks[self.MP_POSE.RIGHT_HIP.value]
        
        if (left_hip.visibility < self.VISIBILITY_THRESH or 
            right_hip.visibility < self.VISIBILITY_THRESH):
            self._reset("Hips not visible")
            return False, "Ensure hips are visible.", None

        # Calculate Angle immediately (needed for dynamic visibility check)
        angle = self._calculate_angle(landmarks)

        # --- STEP 2: DYNAMIC INTEGRITY CHECK (Frame Level) ---
        # Pass 'angle' so we can relax visibility requirements for side profiles (90 deg)
        if not self._check_integrity(landmarks, angle):
            self._reset("User not fully visible")
            return False, "Step back! Make sure full body is in frame.", None

        # --- STEP 3: LOGIC CHECKS ---
        
        # B. Orientation (Angle) Range Check
        if not (self.ANGLE_TOLERANCE[0] < angle < self.ANGLE_TOLERANCE[1]):
            self._reset(f"Bad Angle: {int(angle)}")
            return False, "Rotate to your side (90 deg).", None

        # C. Distance (Scale)
        torso_size = self._calculate_torso(landmarks)
        if torso_size < 0.2:
            self._reset("User too far")
            return False, "Move Closer.", None
        if torso_size > 0.8:
            self._reset("User too close")
            return False, "Step Back.", None

        # --- STEP 4: TEMPORAL BUFFERING ---
        # Frame is valid. Add metrics to the sliding window.
        
        # Capture Heel Y for Floor Baseline
        left_heel_y = landmarks[self.MP_POSE.LEFT_HEEL.value].y
        right_heel_y = landmarks[self.MP_POSE.RIGHT_HEEL.value].y
        avg_heel_y = (left_heel_y + right_heel_y) / 2.0

        frame_metrics = {
            "hip_x": landmarks[self.MP_POSE.LEFT_HIP.value].x,
            "hip_y": landmarks[self.MP_POSE.LEFT_HIP.value].y,
            "angle": angle,
            "torso": torso_size,
            "heel_y": avg_heel_y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        # --- STEP 5: STABILITY CHECK (Window Level) ---
        # We only pass if we have 2 full seconds of data (60 frames)
        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Still... {progress}%", None

        # Buffer is full. Now check if they were actually STILL for those 2 seconds.
        if not self._is_stable():
            # NOTE: We do NOT clear the buffer here. We let it slide.
            return False, "Don't move...", None

        # --- STEP 6: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport()
        return True, "System Ready. Begin!", calibration_data

    def _reset(self, reason):
        """Clears the buffer. User has to start the 2s timer over."""
        # Optional: Print reason for debugging
        # print(f"Gatekeeper Reset: {reason}")
        self.validation_buffer.clear()

    def _check_integrity(self, landmarks, current_angle):
        """
        Checks visibility and clipping. 
        Dynamic: If angle is near 90 (Side), we allow occlusion (1 arm/leg).
        If angle is oblique (near 60 or 120), we require full visibility.
        """
        # 1. Nose is ALWAYS mandatory
        if landmarks[self.MP_POSE.NOSE.value].visibility < self.VISIBILITY_THRESH:
            return False

        # 2. Determine Strictness based on Angle
        # "Side View" zone: 75 to 105 degrees (covers both Left/Right profiles due to abs(angle))
        is_side_profile = (75 < current_angle < 105)

        l_shoulder = landmarks[self.MP_POSE.LEFT_SHOULDER.value]
        r_shoulder = landmarks[self.MP_POSE.RIGHT_SHOULDER.value]
        l_ankle = landmarks[self.MP_POSE.LEFT_ANKLE.value]
        r_ankle = landmarks[self.MP_POSE.RIGHT_ANKLE.value]

        # 3. Check Shoulders
        if is_side_profile:
            # Require at least ONE
            if l_shoulder.visibility < self.VISIBILITY_THRESH and r_shoulder.visibility < self.VISIBILITY_THRESH:
                return False
        else:
            # Require BOTH
            if l_shoulder.visibility < self.VISIBILITY_THRESH or r_shoulder.visibility < self.VISIBILITY_THRESH:
                return False

        # 4. Check Ankles
        if is_side_profile:
            # Require at least ONE
            if l_ankle.visibility < self.VISIBILITY_THRESH and r_ankle.visibility < self.VISIBILITY_THRESH:
                return False
        else:
            # Require BOTH
            if l_ankle.visibility < self.VISIBILITY_THRESH or r_ankle.visibility < self.VISIBILITY_THRESH:
                return False

        # 5. Edge Clipping (Only check what is actually visible)
        # We check all Critical Points that pass the visibility threshold
        points_to_check = [
            self.MP_POSE.NOSE, 
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_ANKLE, self.MP_POSE.RIGHT_ANKLE
        ]
        
        for idx in points_to_check:
            lm = landmarks[idx.value]
            # Only check edge clipping if the point is visible enough to matter
            if lm.visibility > self.VISIBILITY_THRESH:
                if not (0.05 < lm.x < 0.95 and 0.05 < lm.y < 0.95):
                    return False
                    
        return True

    def _calculate_angle(self, landmarks):
        """Calculates facing angle relative to camera."""
        left_hip = landmarks[self.MP_POSE.LEFT_HIP.value]
        right_hip = landmarks[self.MP_POSE.RIGHT_HIP.value]
        
        # Angle of the line connecting hips
        dx = left_hip.x - right_hip.x
        dz = left_hip.z - right_hip.z
        
        angle_rad = np.arctan2(dz, dx)
        angle_deg = np.degrees(angle_rad)
        return abs(angle_deg)

    def _calculate_torso(self, landmarks):
        """Returns torso length as % of screen height."""
        hip_y = landmarks[self.MP_POSE.LEFT_HIP.value].y
        shoulder_y = landmarks[self.MP_POSE.LEFT_SHOULDER.value].y
        return abs(hip_y - shoulder_y)

    def _is_stable(self):
        """Checks variance of Hip X/Y and Angle over the buffered frames."""
        # Extract data from buffer
        hip_x_history = [m['hip_x'] for m in self.validation_buffer]
        hip_y_history = [m['hip_y'] for m in self.validation_buffer]
        angle_history = [m['angle'] for m in self.validation_buffer]
        
        # Calculate Standard Deviation
        std_x = np.std(hip_x_history)
        std_y = np.std(hip_y_history)
        std_angle = np.std(angle_history)
        
        # Combined variance must be low
        if std_x > self.STABILITY_VARIANCE or std_y > self.STABILITY_VARIANCE:
            return False
            
        # Check Angle Consistency (Spinning/Turning)
        if std_angle > self.ANGLE_STABILITY_THRESH:
            return False
            
        return True

    def _generate_passport(self):
        """Averages the buffer to create a robust calibration profile."""
        avg_angle = np.mean([m['angle'] for m in self.validation_buffer])
        avg_torso = np.mean([m['torso'] for m in self.validation_buffer])
        avg_floor = np.mean([m['heel_y'] for m in self.validation_buffer])
        
        return {
            "calibrated_angle": avg_angle,
            "calibrated_scale": avg_torso,
            "floor_y": avg_floor,
            "calibrated_at": time.time()
        }