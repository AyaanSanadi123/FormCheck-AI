import numpy as np

class BicepCurlRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.humerus_l = calibration_data.get('humerus_baseline', 1.0)
        self.elbow_anchor_x = calibration_data.get('elbow_x_anchor', 0.5)
        
        # Thresholds
        self.THRESH_FLEXION_PEAK = 45.0  # Degrees (Lower is more contracted)
        self.THRESH_EXTENSION_FULL = 150.0 # Degrees (Start/End)
        self.THRESH_ELBOW_DRIFT = 0.15   # 15% of humerus length
        self.THRESH_BACK_LEAN = 10.0     # Degrees of spinal deviation
        self.THRESH_DROP_SPEED = 10      # Degrees per frame
        self.THRESH_DIRECTION_CHANGE = 3.0 # Degrees to detect early descent
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.min_angle_achieved = 180 # Peak contraction is the minimum angle
        self.prev_angle = 180

    def process(self, landmarks):
        if not landmarks: return None

        # 1. FETCH JOINTS (Normalized Profile)
        # 12=Sh, 14=Elbow, 16=Wrist, 24=Hip
        sh, elb, wrist, hip = landmarks[12], landmarks[14], landmarks[16], landmarks[24]
        
        # 2. CALCULATE BICEP FLEXION ANGLE
        # Angle between Shoulder-Elbow and Elbow-Wrist
        angle = self._calculate_joint_angle(sh, elb, wrist)

        # 3. CALCULATE BACK LEAN
        # Angle of spine relative to vertical (which the normalizer centered at 0)
        spine_dx = sh.x - hip.x
        spine_dy = sh.y - hip.y
        curr_lean = abs(np.degrees(np.arctan2(spine_dx, -spine_dy)))

        # 4. ELBOW DRIFT (The "Swing" Detector)
        # Difference between current elbow X and the calibrated anchor
        drift = abs(elb.x - self.elbow_anchor_x) / self.humerus_l

        # --- STATE MACHINE ---
        
        if self.state == "IDLE":
            if angle < 140: # Starting the lift
                self._reset_rep(angle)
                self.state = "FLEXING"

        elif self.state == "FLEXING":
            self.min_angle_achieved = min(self.min_angle_achieved, angle)
            
            # FAULT: Elbow Sway (Forward/Backward swing)
            if drift > self.THRESH_ELBOW_DRIFT:
                self._add_fault("ELBOW_SWAY", 15, "Keep Elbows Pinned")

            # FAULT: Back Lean (Using momentum)
            if curr_lean > self.THRESH_BACK_LEAN:
                self._add_fault("BACK_LEAN", 10, "Stay Upright - No Swinging")

            # TRANSITION 1: Target Peak Reached
            if angle < self.THRESH_FLEXION_PEAK:
                self.state = "EXTENDING"
            
            # TRANSITION 2: Early Descent (Directional Switch)
            # If the user starts lowering before reaching THRESH_FLEXION_PEAK
            elif (angle - self.prev_angle) > self.THRESH_DIRECTION_CHANGE:
                self._add_fault("PARTIAL_ROM", 15, "Squeeze Higher at Top")
                self.state = "EXTENDING"

        elif self.state == "EXTENDING":
            # Check for uncontrolled drop speed
            if self._is_dropping(angle): # Check for uncontrolled drop
                 self._add_fault("CONTROL", 10, "Lower Slowly")

            # Finalize Rep
            if angle > self.THRESH_EXTENSION_FULL:
                self._finalize_rep_success()
                self.state = "IDLE"
        
        self.prev_angle = angle
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "angle": int(angle),
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks
        }

    def _calculate_joint_angle(self, p1, p2, p3):
        """Calculates the angle at the elbow (p2)."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        
        dot_prod = np.dot(unit_v1, unit_v2)
        return np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

    def _is_dropping(self, current_angle):
        return (current_angle - self.prev_angle) > self.THRESH_DROP_SPEED

    def _reset_rep(self, current_angle):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.prev_angle = current_angle
        self.min_angle_achieved = 180

    def _add_fault(self, code, penalty, msg):
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finalize_rep_success(self):
        # Rep only counts if score is salvageable and some effort was made
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback_buffer = "Rep Failed - Form too poor"