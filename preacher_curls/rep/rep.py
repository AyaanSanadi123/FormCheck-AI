import numpy as np
import time

class RepLogic:
    def __init__(self, calibration_data):
        """
        Initializes Preacher Curl Rep Logic.
        Standard: Up-First (IDLE -> CONCENTRIC -> TOP -> ECCENTRIC -> COMPLETE).
        """
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.SCORE_MAX = 100
        self.current_score = self.SCORE_MAX
        self.feedback = "Ready"
        self.faults = [] # Store as list of dicts for consistency
        
        # Thresholds (Degrees)
        self.THRESH_ROM_BOTTOM = 160.0  # Full extension requirement
        self.THRESH_ROM_TOP = 45.0     # Peak contraction requirement
        self.THRESH_ELBOW_LIFT = 0.15  # Normalized Y-drift of shoulder relative to elbow
        
        # Tracking
        self.prev_angle = 180.0
        self.prev_time = None
        self.angular_velocity = 0.0 # Added for consistency

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Processes standardized landmarks to track preacher curl form.
        """
        if not landmarks:
            return None

        # Handle initial prev_time setup
        if self.prev_time is None:
            self.prev_time = timestamp if timestamp is not None else time.time()
            self.prev_angle = self._calculate_angle(landmarks[12], landmarks[14], landmarks[16]) # Initialize prev_angle for first dt
            # Return an initial packet, not a full logic run
            return {
                "state": self.state,
                "reps": self.rep_count,
                "score": self.current_score,
                "feedback": self.feedback,
                "faults": [f['code'] for f in self.faults],
                "coords": landmarks,
                "raw_coords": raw_landmarks,
                "metrics": {"angle": int(self.prev_angle), "velocity": 0} # Initial values
            }

        # 1. Setup Data
        # Coordinates: Shoulder(12), Elbow(14), Wrist(16)
        # In normalized space, Elbow is (0,0).
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        curr_time = timestamp if timestamp else time.time()
        dt = max(curr_time - self.prev_time, 0.001)
        
        # Calculate Angle
        angle = self._calculate_angle(shoulder, elbow, wrist)
        self.angular_velocity = (angle - self.prev_angle) / dt

        # 2. State Machine (Up-First)
        if self.state == "IDLE":
            # Start rep when angle begins decreasing (curling up)
            if angle < 155 and self.angular_velocity < -10:
                self._reset_rep()
                self.state = "CONCENTRIC"
                self.feedback = "Curl up!"

        elif self.state == "CONCENTRIC":
            # Fault Check: Elbow Lift (Shoulder moving too much indicates pad separation)
            if abs(shoulder.y + 1.0) > self.THRESH_ELBOW_LIFT: # Shoulder should be ~ -1.0 relative to Elbow
                self._add_fault("ELBOW_LIFT", 15, "Keep arms flush against the pad")

            if angle < self.THRESH_ROM_TOP:
                self.state = "TOP"
                self.feedback = "Squeeze and hold!"

        elif self.state == "TOP":
            # Transition to eccentric when arm starts dropping
            if self.angular_velocity > 10:
                self.state = "ECCENTRIC"
                self.feedback = "Lower slowly"

        elif self.state == "ECCENTRIC":
            # Check for "Short-Reps" (User stops before full extension)
            if angle > self.THRESH_ROM_BOTTOM and abs(self.angular_velocity) < 5: # Added velocity check
                self.state = "COMPLETE"
            elif self.angular_velocity < -5 and angle > 100: # Prematurely starting next rep
                 self._add_fault("SHORT_ROM", 20, "Extend fully to the bottom")
                 self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finalize_rep_success()
            self.state = "IDLE"

        # Update trackers
        self.prev_angle = angle
        self.prev_time = curr_time

        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback,
            "faults": [f['code'] for f in self.faults], # Return only codes as list of strings
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"angle": angle, "velocity": self.angular_velocity}
        }

    def _calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm == 0: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)))

    def _reset_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per rep."""
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg}) # Store as dictionary
        self.feedback = msg # Always update feedback for the latest fault

    def _finalize_rep_success(self):
        """Finalizes the rep with nuanced feedback."""
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)" # Standard feedback
        else:
            self.feedback = "Rep Failed - Form too poor"