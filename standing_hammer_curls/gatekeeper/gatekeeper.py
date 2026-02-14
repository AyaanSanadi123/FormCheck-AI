import numpy as np
import time

class HammerCurlGatekeeper:
    def __init__(self):
        self.CONSISTENCY_WINDOW = 45
        self.REQUIRED_VISIBILITY = 0.90
        self.is_calibrated = False
        self.calibration_data = {}
        self.window_buffer = []
        self.feedback = "Stand sideways with a neutral (hammer) grip"

    def process(self, landmarks):
        if not landmarks or len(landmarks) < 33:
            return False

        # Essential Side-Profile Joints
        sh, elb, wrist = landmarks[12], landmarks[14], landmarks[16]
        index = landmarks[20] # Used to check grip rotation
        
        # 1. Visibility Check
        if any(lm.visibility < self.REQUIRED_VISIBILITY for lm in [sh, elb, wrist]):
            self.feedback = "Side profile obscured"
            return False

        # 2. Neutral Grip Check (Hammer Logic)
        # In a side profile, a neutral grip means the thumb/index 
        # is roughly at the same Z-depth as the wrist.
        grip_rotation = abs(wrist.z - index.z)
        if grip_rotation > 0.05:
            self.feedback = "Keep palms facing your body (Neutral Grip)"
            return False

        # 3. Capture Proportions
        current_metrics = {
            'humerus_l': np.linalg.norm([sh.x - elb.x, sh.y - elb.y]),
            'elbow_x_anchor': elb.x,
            'spine_y_dist': abs(sh.y - landmarks[24].y)
        }
        self.window_buffer.append(current_metrics)

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            self._finalize_calibration()
            return True

        return False

    def _finalize_calibration(self):
        self.calibration_data = {
            'humerus_baseline': np.mean([f['humerus_l'] for f in self.window_buffer]),
            'elbow_x_anchor': np.mean([f['elbow_x_anchor'] for f in self.window_buffer]),
            'exercise_id': 'hammer_curl'
        }
        self.is_calibrated = True
        self.feedback = "Hammer Curls: Keep wrists stiff!"