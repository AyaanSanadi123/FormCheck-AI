import numpy as np
import mediapipe as mp
import time
from collections import deque

class SeatedRowGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) 
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.005 # Variance in wrist position
        
        # Biomechanical Tolerances
        self.MIN_TORSO_ANGLE = 65.0   # Degrees (90 is perfectly upright)
        self.MAX_TORSO_ANGLE = 115.0  # Degrees
        
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
            self._reset()
            return False, "Looking for lifter...", None

        # --- STEP 1: VISIBILITY CHECK ---
        # We don't strictly require ankles because machine footplates often block them.
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP,
            self.MP_POSE.LEFT_KNEE, self.MP_POSE.RIGHT_KNEE,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset()
                return False, "Ensure Torso & Arms Visible", None

        # --- STEP 2: CALCULATE JOINT AVERAGES ---
        # Averages provide robustness regardless of facing direction
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        
        avg_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        avg_knee_x = (landmarks[25].x + landmarks[26].x) / 2
        avg_knee_y = (landmarks[25].y + landmarks[26].y) / 2
        
        avg_wrist_x = (landmarks[15].x + landmarks[16].x) / 2
        avg_wrist_y = (landmarks[15].y + landmarks[16].y) / 2

        # --- STEP 3: DETERMINE FACING DIRECTION & SEATED CHECK ---
        # Knees must be horizontally in front of the hips.
        # We determine facing side based on where knees are relative to hips.
        # If knees_x > hips_x, facing Right (1). If knees_x < hips_x, facing Left (-1).
        if avg_knee_x > avg_hip_x:
            facing_dir = 1.0
        else:
            facing_dir = -1.0
            
        # Check if knees are actually far enough away (seated with legs forward)
        if abs(avg_knee_x - avg_hip_x) < 0.1:
            self._reset()
            return False, "Sit Down & Place Feet", None

        # --- STEP 4: THE TORSO CHECK (UPRIGHT POSTURE) ---
        # Calculate angle of torso relative to horizontal floor
        # 90 degrees is perfectly upright.
        # < 90 is leaning forward (towards knees).
        # > 90 is leaning backward (away from knees).
        
        # Vector from Hip to Shoulder
        vec_x = avg_sh_x - avg_hip_x
        vec_y = avg_sh_y - avg_hip_y # Y increases downwards in MP
        
        # Invert Y because MP has inverted Y-axis (top is 0)
        # We want +Y to be UP for standard angle calculation
        vec_y = -vec_y 
        
        # Adjust X based on facing direction to standardize
        # If facing right, forward is +X. If facing left, forward is -X (so we flip it).
        vec_x = vec_x * facing_dir
        
        # Calculate angle in degrees (0 to 180)
        # atan2(y, x) returns angle from positive X axis
        # specific to this setup:
        # if upright: x=0, y>0 -> 90 deg
        # if leaning forward: x>0, y>0 -> < 90 deg (wait, standard trig: 0 is right, 90 is up, 180 is left)
        # Correction: 
        # Facing Right: Shoulder is LEFT of Hip (Forward Lean) -> x < 0. Wait.
        # Let's visualize: 
        # Hips at (0,0). Knees at (1,0) (Facing Right).
        # Shoulders at (-0.2, 1) (Leaning Back).
        # vec_x (raw) = -0.2. vec_x (adj) = -0.2 * 1 = -0.2.
        # angle = atan2(1, -0.2) = ~101 deg (Obtuse, correct).
        # Shoulders at (0.2, 1) (Leaning Forward).
        # vec_x (raw) = 0.2. vec_x (adj) = 0.2.
        # angle = atan2(1, 0.2) = ~78 deg (Acute, correct).
        
        torso_angle = np.degrees(np.arctan2(vec_y, vec_x))
        
        # Normalize to 0-180 just in case
        if torso_angle < 0: torso_angle += 360
        
        if torso_angle < self.MIN_TORSO_ANGLE or torso_angle > self.MAX_TORSO_ANGLE:
            self._reset()
            return False, f"Sit Upright ({int(torso_angle)}°)", None

        # --- STEP 5: THE REACH CHECK (ARMS FORWARD) ---
        # The wrists must be significantly in front of the shoulders to indicate starting position.
        torso_length = np.sqrt((avg_sh_x - avg_hip_x)**2 + (avg_sh_y - avg_hip_y)**2)
        reach_distance = abs(avg_wrist_x - avg_sh_x)
        
        if reach_distance < (torso_length * 0.6):
            self._reset()
            return False, "Straighten Arms (Reach Forward)", None

        # --- STEP 6: STABILITY BUFFER ---
        frame_metrics = {
            "torso_angle": torso_angle,
            "wrist_x": avg_wrist_x,
            "hip_x": avg_hip_x,
            "hip_y": avg_hip_y,
            "sh_x": avg_sh_x,
            "sh_y": avg_sh_y,
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Stretch... {progress}%", None

        if not self._is_stable():
            return False, "Hold Still...", None

        # --- STEP 7: SUCCESS / CALIBRATION ---
        # Use the already determined facing_dir
        calibration_data = self._generate_passport(landmarks, facing_dir)
        return True, "ROW READY!", calibration_data

    def _reset(self):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if the wrists and torso have been still using Standard Deviation."""
        wrist_history = [m['wrist_x'] for m in self.validation_buffer]
        # Using standard deviation is more intuitive than variance
        # Threshold: 0.02 normalized units (approx 2% of screen width movement allowed)
        if np.std(wrist_history) > 0.02: 
            return False
        return True

    def _generate_passport(self, landmarks, facing_dir):
        """Captures user dimensions and setup state."""
        # 2. Capture Setup Torso Angle (For the 'Momentum Swing' fault later)
        setup_torso_angle = np.mean([m['torso_angle'] for m in self.validation_buffer])
        
        # 3. Hip Origin (To zero the Normalizer grid)
        hip_origin_x = np.mean([m['hip_x'] for m in self.validation_buffer])
        hip_origin_y = np.mean([m['hip_y'] for m in self.validation_buffer])
        
        # 4. Torso Length (Scale factor)
        avg_sh_x = np.mean([m['sh_x'] for m in self.validation_buffer])
        avg_sh_y = np.mean([m['sh_y'] for m in self.validation_buffer])
        torso_length = np.sqrt((avg_sh_x - hip_origin_x)**2 + (avg_sh_y - hip_origin_y)**2)
        
        return {
            "facing_side": facing_dir,
            "hip_origin_x": hip_origin_x,
            "hip_origin_y": hip_origin_y,
            "setup_torso_angle": setup_torso_angle,
            "torso_length": torso_length,
            "calibrated_at": time.time()
        }