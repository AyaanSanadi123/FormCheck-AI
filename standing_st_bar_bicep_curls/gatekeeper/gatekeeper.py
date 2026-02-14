import numpy as np
import time

class BicepCurlGatekeeper:
    def __init__(self):
        # --- CONFIGURATION ---
        self.CONSISTENCY_WINDOW = 45  # ~1.5 seconds at 30 FPS
        self.REQUIRED_VISIBILITY = 0.90
        
        # State Management
        self.is_calibrated = False
        self.calibration_data = {}
        self.window_buffer = []
        
        # Feedback for UI
        self.feedback = "Stand sideways to the camera with arms fully extended"

    def process(self, landmarks):
        """
        Validates side-profile posture and latches:
        1. Humerus Length (Shoulder to Elbow)
        2. Forearm Length (Elbow to Wrist)
        3. Neutral Spine Angle (Shoulder to Hip)
        """
        if not landmarks or len(landmarks) < 33:
            return False

        # 1. Profile Joints (Right Side - IDs 12, 14, 16, 24, 26, 28)
        # We focus on the side closest to the camera.
        essential_indices = [12, 14, 16, 24] 
        
        # 2. Frame-Level Rejection: Visibility
        for idx in essential_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.feedback = "Ensure your side profile is visible"
                self.window_buffer = []
                return False

        # 3. Frame-Level Rejection: Profile Orientation
        # Calculate the 'depth' (Z) difference between shoulders.
        # In a true profile, one shoulder should be significantly 'behind' the other.
        z_depth_profile = abs(landmarks[11].z - landmarks[12].z)
        if z_depth_profile < 0.25:
            self.feedback = "Turn 90 degrees to face the side"
            self.window_buffer = []
            return False

        # 4. Frame-Level Rejection: Initial State (Arms Down)
        # The wrist must be lower than the elbow for a valid start.
        if landmarks[16].y < landmarks[14].y:
            self.feedback = "Start with the bar at your thighs"
            self.window_buffer = []
            return False

        # 5. Stability Check & Anatomical Metric Gathering
        current_metrics = {
            'humerus_l': self._dist(landmarks[12], landmarks[14]),
            'forearm_l': self._dist(landmarks[14], landmarks[16]),
            'spine_vector': [landmarks[12].x - landmarks[24].x, landmarks[12].y - landmarks[24].y],
            'elbow_anchor_x': landmarks[14].x
        }
        self.window_buffer.append(current_metrics)

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            self._finalize_calibration()
            return True

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        self.feedback = f"Calibrating Side Profile... {progress}%"
        return False

    def _dist(self, p1, p2):
        """Standard Euclidean distance for 2D profile segments."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _finalize_calibration(self):
        """Averages the buffer to create the Calibration Passport."""
        self.calibration_data = {
            'humerus_baseline': np.mean([f['humerus_l'] for f in self.window_buffer]),
            'forearm_baseline': np.mean([f['forearm_l'] for f in self.window_buffer]),
            'neutral_spine_v': np.mean([f['spine_vector'] for f in self.window_buffer], axis=0).tolist(),
            'elbow_x_anchor': np.mean([f['elbow_x_anchor'] for f in self.window_buffer]),
            'exercise_id': 'standing_barbell_curl',
            'calibrated_at': time.time()
        }
        self.is_calibrated = True
        self.feedback = "Ready! Keep your elbows pinned."

    def get_data(self):
        return self.calibration_data