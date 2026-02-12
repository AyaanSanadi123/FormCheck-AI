import numpy as np
import mediapipe as mp
import time
from collections import deque

class DeadliftGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.REQUIRED_DURATION = 1.5  # Seconds (Deadlift setup is faster than bench)
        self.BUFFER_SIZE = int(self.FPS * self.REQUIRED_DURATION) # 45 Frames
        
        # Thresholds
        self.VISIBILITY_THRESH = 0.65
        self.STABILITY_VARIANCE = 0.005 # For Hips (Must be still)
        self.ARM_VERTICAL_TOLERANCE = 0.15 # Arms must hang somewhat straight
        
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

        # --- STEP 1: VISIBILITY CHECK (Full Chain) ---
        # Deadlift needs Feet(27-32), Hips(23,24), Shoulders(11,12), Wrists(15,16)
        # We check primarily the side closer to camera later, but need basics now.
        critical_indices = [
            self.MP_POSE.LEFT_SHOULDER, self.MP_POSE.RIGHT_SHOULDER,
            self.MP_POSE.LEFT_HIP, self.MP_POSE.RIGHT_HIP,
            self.MP_POSE.LEFT_KNEE, self.MP_POSE.RIGHT_KNEE,
            self.MP_POSE.LEFT_ANKLE, self.MP_POSE.RIGHT_ANKLE,
            self.MP_POSE.LEFT_WRIST, self.MP_POSE.RIGHT_WRIST
        ]
        
        for idx in critical_indices:
            if landmarks[idx.value].visibility < self.VISIBILITY_THRESH:
                self._reset(f"Joint {idx.name} hidden")
                return False, "Full Body Visible?", None

        # --- STEP 2: TRIANGLE HIERARCHY CHECK ---
        # Rule: Shoulders > Hips > Knees (In height/elevation)
        # MediaPipe Y: 0 is Top, 1 is Bottom.
        # So: Shoulder Y < Hip Y < Knee Y
        
        l_sh = landmarks[11]
        l_hip = landmarks[23]
        l_knee = landmarks[25]
        
        # We use average of Left/Right to be robust
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        avg_knee_y = (landmarks[25].y + landmarks[26].y) / 2
        
        # Check 1: Are Hips below Shoulders?
        if avg_sh_y >= avg_hip_y:
            self._reset("Hips too high")
            return False, "Lower your Hips", None
            
        # Check 2: Are Knees below Hips?
        if avg_hip_y >= avg_knee_y:
            self._reset("Squatting too deep")
            return False, "Raise Hips (Don't Squat)", None

        # --- STEP 2.5: SHIN VERTICALITY CHECK ---
        # Shins must be near-vertical (bar over mid-foot).
        # Horizontal distance between Knee and Ankle should be minimal.
        avg_knee_x = (landmarks[25].x + landmarks[26].x) / 2
        avg_ankle_x = (landmarks[27].x + landmarks[28].x) / 2
        
        # Using a tolerance similar to arm verticality
        if abs(avg_knee_x - avg_ankle_x) > 0.10: 
            self._reset("Shins too angled")
            return False, "Shins Vertical (Hips Back)", None

        # --- STEP 3: ARM VERTICALITY CHECK ---
        # Arms should hang straight down in setup. 
        # X-distance between Shoulder and Wrist should be small.
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_wrist_x = (landmarks[15].x + landmarks[16].x) / 2
        
        if abs(avg_sh_x - avg_wrist_x) > self.ARM_VERTICAL_TOLERANCE:
            self._reset("Arms not vertical")
            return False, "Arms Straight Down", None

        # --- STEP 4: STABILITY BUFFER ---
        # We track Hip Y to ensure they are holding the "Wedge"
        frame_metrics = {
            "hip_y": avg_hip_y,
            "ankle_y": (landmarks[27].y + landmarks[28].y) / 2, # Floor height
            "timestamp": time.time()
        }
        self.validation_buffer.append(frame_metrics)

        if len(self.validation_buffer) < self.BUFFER_SIZE:
            progress = int((len(self.validation_buffer) / self.BUFFER_SIZE) * 100)
            return False, f"Hold Position... {progress}%", None

        if not self._is_stable():
            return False, "Stay Still...", None

        # --- STEP 5: SUCCESS / CALIBRATION ---
        calibration_data = self._generate_passport(landmarks)
        return True, "DEADLIFT READY!", calibration_data

    def _reset(self, reason):
        self.validation_buffer.clear()

    def _is_stable(self):
        """Checks if Hips have been still."""
        hip_history = [m['hip_y'] for m in self.validation_buffer]
        if np.var(hip_history) > self.STABILITY_VARIANCE:
            return False
        return True

    def _generate_passport(self, landmarks):
        """Captures user dimensions."""
        # Calculate Floor Level (Average Ankle Y)
        floor_y = np.mean([m['ankle_y'] for m in self.validation_buffer])
        
        # Calculate Torso Length (Shoulder to Hip)
        avg_sh_x = (landmarks[11].x + landmarks[12].x) / 2
        avg_sh_y = (landmarks[11].y + landmarks[12].y) / 2
        avg_hip_x = (landmarks[23].x + landmarks[24].x) / 2
        avg_hip_y = (landmarks[23].y + landmarks[24].y) / 2
        
        torso_length = np.sqrt((avg_sh_x - avg_hip_x)**2 + (avg_sh_y - avg_hip_y)**2)
        
        # Determine Facing Side (Left or Right)
        # If Nose X < Hip X, likely facing Left (assuming standard setup)
        # Better: Use the side-facing logic from Bench Gatekeeper
        facing_side = self._detect_facing_side(landmarks)
        
        return {
            "floor_y": floor_y,
            "torso_length": torso_length,
            "facing_side": facing_side,
            "calibrated_at": time.time()
        }

    def _detect_facing_side(self, landmarks):
        """Returns -1 for Left, 1 for Right."""
        # Standard side-view heuristic: Nose X vs Mid-Hip X
        # If Nose is to the left of Hips, user is facing Left (-1).
        # If Nose is to the right of Hips, user is facing Right (1).
        
        nose_x = landmarks[0].x
        hip_center_x = (landmarks[23].x + landmarks[24].x) / 2
        
        if nose_x < hip_center_x:
            return -1 # Facing Left
        return 1 # Facing Right