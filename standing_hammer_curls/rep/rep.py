import numpy as np

class HammerCurlRep:
    def __init__(self, calibration_data):
        # Safety check: avoid division by zero
        self.humerus_l = calibration_data.get('humerus_baseline', 1.0)
        if self.humerus_l == 0: self.humerus_l = 1.0
            
        self.elbow_x_anchor = calibration_data.get('elbow_x_anchor', 0.5)
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = 100
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Thresholds
        self.THRESH_PEAK = 50.0        # Ideal contraction
        self.THRESH_EXTENSION_FULL = 150.0
        self.THRESH_SWING = 0.12       # 12% of humerus length
        self.THRESH_GRIP_DRIFT = 0.08  # Max Z-variance for neutral grip
        self.THRESH_DROP_SPEED = 10    # Degrees per frame
        self.THRESH_DIRECTION_CHANGE = 3.0 # Degrees to detect early descent

        # Tracking
        self.prev_angle = 180
        self.min_angle_achieved = 180

    def process(self, landmarks):
        if not landmarks or len(landmarks) < 33: return None

        # 1. FETCH JOINTS (Side Profile: 12=Sh, 14=Elb, 16=Wrist)
        sh, elb, wrist = landmarks[12], landmarks[14], landmarks[16]
        thumb, pinky = landmarks[21], landmarks[18]
        
        # 2. FLEXION ANGLE
        angle = self._get_angle(sh, elb, wrist)
        
        # 3. ELBOW DRIFT (Normalized)
        drift = abs(elb.x - self.elbow_x_anchor) / self.humerus_l
        
        # 4. GRIP ROTATION (Z-Depth variance)
        grip_drift = abs(thumb.z - pinky.z)

        # --- STATE MACHINE ---
        if self.state == "IDLE":
            if angle < 140:
                self._reset_rep(angle)
                self.state = "CURLING"

        elif self.state == "CURLING":
            self.min_angle_achieved = min(self.min_angle_achieved, angle)
            
            # Monitoring Form
            if drift > self.THRESH_SWING:
                self._add_fault("ELBOW_SWING", 15, "Keep Elbows Fixed")
            
            if grip_drift > self.THRESH_GRIP_DRIFT:
                self._add_fault("GRIP_ROTATE", 10, "Keep Palms Facing Body")
            
            # TRANSITION 1: Target Peak Reached
            if angle < self.THRESH_PEAK:
                self.state = "DESCENDING"
            
            # TRANSITION 2: Early Descent (Directional Switch)
            # If the user starts lowering before reaching THRESH_PEAK
            elif (angle - self.prev_angle) > self.THRESH_DIRECTION_CHANGE:
                self._add_fault("PARTIAL_ROM", 15, "Squeeze Higher at Top")
                self.state = "DESCENDING"

        elif self.state == "DESCENDING":
            # Check for uncontrolled drop speed
            if self._is_dropping(angle):
                 self._add_fault("CONTROL", 10, "Lower Slowly")

            # Finalize Rep
            if angle > self.THRESH_EXTENSION_FULL:
                self._finalize_rep_success()
                self.state = "IDLE"

        self.prev_angle = angle
        return {
            "reps": self.rep_count,
            "state": self.state,
            "score": self.current_score,
            "angle": int(angle),
            "faults": list(set([f['code'] for f in self.faults])),
            "feedback": self.feedback_buffer,
            "coords": landmarks
        }

    def _get_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))

    def _is_dropping(self, current_angle):
        return (current_angle - self.prev_angle) > self.THRESH_DROP_SPEED

    def _reset_rep(self, current_angle):
        self.current_score = 100
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