import numpy as np
import time
import mediapipe as mp

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

        # MediaPipe Indices
        self.MP_POSE = mp.solutions.pose.PoseLandmark

    def check(self, landmarks): # Renamed from process to check
        """
        Validates side-profile posture and latches:
        1. Humerus Length (Shoulder to Elbow)
        2. Forearm Length (Elbow to Wrist)
        3. Neutral Spine Angle (Shoulder to Hip)
        """
        if not landmarks or len(landmarks) < 33:
            self.feedback = "No landmarks detected"
            return False, self.feedback, None

        # 1. Profile Joints (Left or Right side, determined by Z-depth)
        # Use both shoulders and hips to ensure at least one side is fully visible
        l_sh, r_sh = landmarks[self.MP_POSE.LEFT_SHOULDER.value], landmarks[self.MP_POSE.RIGHT_SHOULDER.value]
        l_elb, r_elb = landmarks[self.MP_POSE.LEFT_ELBOW.value], landmarks[self.MP_POSE.RIGHT_ELBOW.value]
        l_wrist, r_wrist = landmarks[self.MP_POSE.LEFT_WRIST.value], landmarks[self.MP_POSE.RIGHT_WRIST.value]
        l_hip, r_hip = landmarks[self.MP_POSE.LEFT_HIP.value], landmarks[self.MP_POSE.RIGHT_HIP.value]

        # Determine active side by which shoulder is closer to camera (smaller Z)
        active_side_is_left = l_sh.z < r_sh.z

        # Define essential indices for the active side
        if active_side_is_left:
            essential_indices = [
                self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_ELBOW.value,
                self.MP_POSE.LEFT_WRIST.value, self.MP_POSE.LEFT_HIP.value
            ]
            active_shoulder = l_sh
            active_elbow = l_elb
            active_wrist = l_wrist
            active_hip = l_hip
        else:
            essential_indices = [
                self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_ELBOW.value,
                self.MP_POSE.RIGHT_WRIST.value, self.MP_POSE.RIGHT_HIP.value
            ]
            active_shoulder = r_sh
            active_elbow = r_elb
            active_wrist = r_wrist
            active_hip = r_hip
        
        # 2. Frame-Level Rejection: Visibility
        for idx in essential_indices:
            if landmarks[idx].visibility < self.REQUIRED_VISIBILITY:
                self.feedback = f"Ensure active side ({'Left' if active_side_is_left else 'Right'}) is visible"
                self.window_buffer = []
                return False, self.feedback, None

        # 3. Frame-Level Rejection: Profile Orientation
        # Check Z-depth difference between shoulders. Smaller diff means more frontal.
        z_depth_profile = abs(l_sh.z - r_sh.z)
        if z_depth_profile < 0.25: 
            self.feedback = "Turn 90 degrees to face the side"
            self.window_buffer = []
            return False, self.feedback, None

        # 4. Frame-Level Rejection: Initial State (Arms Down)
        # The wrist must be lower than the elbow for a valid start.
        if active_wrist.y < active_elbow.y or active_wrist.y < active_shoulder.y:
            self.feedback = "Start with the bar at your thighs, arms extended"
            self.window_buffer = []
            return False, self.feedback, None

        # 5. Stability Check & Anatomical Metric Gathering
        current_metrics = {
            'humerus_l': self._dist(active_shoulder, active_elbow),
            'forearm_l': self._dist(active_elbow, active_wrist),
            'spine_vector_x': active_shoulder.x - active_hip.x,
            'spine_vector_y': active_shoulder.y - active_hip.y,
            'elbow_anchor_x': active_elbow.x
        }
        self.window_buffer.append(current_metrics)

        if len(self.window_buffer) >= self.CONSISTENCY_WINDOW:
            self._finalize_calibration(active_side_is_left)
            return True, self.feedback, self.calibration_data

        progress = int((len(self.window_buffer) / self.CONSISTENCY_WINDOW) * 100)
        self.feedback = f"Calibrating Side Profile... {progress}%"
        return False, self.feedback, None

    def _dist(self, p1, p2):
        """Standard Euclidean distance for 2D profile segments."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _finalize_calibration(self, active_side_is_left):
        """Averages the buffer to create the Calibration Passport."""
        active_side = "LEFT" if active_side_is_left else "RIGHT"

        humerus_baseline = np.mean([f['humerus_l'] for f in self.window_buffer])
        forearm_baseline = np.mean([f['forearm_l'] for f in self.window_buffer])
        spine_vec_x = np.mean([f['spine_vector_x'] for f in self.window_buffer])
        spine_vec_y = np.mean([f['spine_vector_y'] for f in self.window_buffer])


        self.calibration_data = {
            'active_side': active_side,
            'scale_factor': humerus_baseline, # Using humerus length as scale factor
            'humerus_baseline': humerus_baseline, # Keep for rep logic
            'forearm_baseline': forearm_baseline, # Keep for rep logic
            'neutral_spine_vector': [spine_vec_x, spine_vec_y],
            'elbow_x_anchor': np.mean([f['elbow_anchor_x'] for f in self.window_buffer]),
            'exercise_id': 'standing_barbell_curl',
            'calibrated_at': time.time()
        }
        self.is_calibrated = True
        self.feedback = "Ready! Keep your elbows pinned."