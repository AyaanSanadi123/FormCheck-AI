import numpy as np
import time
import mediapipe as mp # Added for MP_POSE indices

class HammerCurlGatekeeper:
    def __init__(self):
        self.CONSISTENCY_WINDOW = 45
        self.REQUIRED_VISIBILITY = 0.90
        self.is_calibrated = False
        self.calibration_data = {}
        self.window_buffer = []
        self.feedback = "Stand sideways with a neutral (hammer) grip"
        self.MP_POSE = mp.solutions.pose.PoseLandmark # Added for MP_POSE indices

    def check(self, landmarks): # Renamed from process to check
        if not landmarks or len(landmarks) < 33:
            self.feedback = "No landmarks detected"
            return False, self.feedback, None

        # Essential Side-Profile Joints
        sh = landmarks[self.MP_POSE.RIGHT_SHOULDER.value] # Using right shoulder for side view
        elb = landmarks[self.MP_POSE.RIGHT_ELBOW.value]
        wrist = landmarks[self.MP_POSE.RIGHT_WRIST.value]
        index = landmarks[self.MP_POSE.RIGHT_INDEX.value] # Used to check grip rotation
        
        # 1. Visibility Check
        if any(lm.visibility < self.REQUIRED_VISIBILITY for lm in [sh, elb, wrist, index]):
            self.feedback = "Side profile obscured or joints not visible"
            self.window_buffer = []
            return False, self.feedback, None

        # 2. Neutral Grip Check (Hammer Logic)
        grip_rotation = abs(wrist.z - index.z)
        if grip_rotation > 0.05:
            self.feedback = "Keep palms facing your body (Neutral Grip)"
            self.window_buffer = []
            return False, self.feedback, None

        # 3. Capture Proportions
        current_metrics = {
            'humerus_l': np.linalg.norm([sh.x - elb.x, sh.y - elb.y]),
            'elbow_x_anchor': elb.x,
            'spine_y_dist': abs(sh.y - landmarks[self.MP_POSE.RIGHT_HIP.value].y) # Use right hip
        }
        self.window_buffer.append(current_metrics)

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            self._finalize_calibration(landmarks) # Pass landmarks to determine active_side
            return True, self.feedback, self.calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        self.feedback = f"Analyzing Posture... {progress}%"
        return False, self.feedback, None

    def _finalize_calibration(self, landmarks):
        # Determine active side
        l_shoulder_z = landmarks[self.MP_POSE.LEFT_SHOULDER.value].z
        r_shoulder_z = landmarks[self.MP_POSE.RIGHT_SHOULDER.value].z
        active_side = "LEFT" if l_shoulder_z < r_shoulder_z else "RIGHT"

        humerus_baseline = np.mean([f['humerus_l'] for f in self.window_buffer])

        self.calibration_data = {
            'active_side': active_side,
            'scale_factor': humerus_baseline, # Using humerus length as scale factor
            'elbow_x_anchor': np.mean([f['elbow_x_anchor'] for f in self.window_buffer]),
            'exercise_id': 'hammer_curl',
            'calibrated_at': time.time()
        }
        self.is_calibrated = True
        self.feedback = "Hammer Curls: Keep wrists stiff!"