import numpy as np
import time

class RepLogic:
    def __init__(self, calibration_data):
        """
        Initializes Tricep Extension (Pushdown) Rep Logic.
        Standard: Down-First (IDLE -> CONCENTRIC -> BOTTOM -> ECCENTRIC -> COMPLETE).
        Note: For pushdowns, 'Concentric' is the downward push.
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
        self.THRESH_ROM_TOP = 60.0      # Deep stretch at the top
        self.THRESH_ROM_LOCKOUT = 160.0 # Requirement for full extension
        self.THRESH_SHOULDER_DRIVE = 0.2 # Normalized Y-drift of shoulder
        
        # Tracking
        self.prev_angle = 60.0
        self.prev_time = time.time() # Initialized for dt calculation
        self.angular_velocity = 0.0 # Added for consistency

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Processes standardized landmarks to track tricep extension quality.
        """
        if not landmarks:
            return None

        # 1. Setup Data
        # Coordinates: Shoulder(12), Elbow(14), Wrist(16)
        # In normalized space, Elbow is (0,0). Shoulder is at approx (0, -1).
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        curr_time = timestamp if timestamp else time.time()
        dt = max(curr_time - self.prev_time, 0.001) # Avoid division by zero
        
        # Calculate Angle
        angle = self._calculate_angle(shoulder, elbow, wrist)
        self.angular_velocity = (angle - self.prev_angle) / dt

        # 2. State Machine (Down-First)
        if self.state == "IDLE":
            # Start rep when user begins pushing down (angle increasing)
            if angle > 70 and self.angular_velocity > 15:
                self._reset_rep()
                self.state = "CONCENTRIC"
                self.feedback = "Push down to lockout!"

        elif self.state == "CONCENTRIC":
            # Fault Check: Shoulder Drive (Elbow moving forward/down)
            # If shoulder-y drifts significantly, they are 'pressing' the weight.
            if abs(shoulder.y + 1.0) > self.THRESH_SHOULDER_DRIVE:
                self._add_fault("SHOULDER_DRIVE", 15, "Keep elbows pinned to your sides!")

            if angle > self.THRESH_ROM_LOCKOUT:
                self.state = "BOTTOM"
                self.feedback = "Squeeze the triceps!"

        elif self.state == "BOTTOM":
            # Transition to eccentric when arm starts rising
            if self.angular_velocity < -15:
                self.state = "ECCENTRIC"
                self.feedback = "Control the rise"

        elif self.state == "ECCENTRIC":
            # Check for "Partial Reps" at the top
            if angle < self.THRESH_ROM_TOP:
                self.state = "COMPLETE"
            elif self.angular_velocity > 10 and angle < 100: # Starting next rep too early
                 self._add_fault("SHORT_ROM", 20, "Bring the bar higher for a stretch")
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
        if self.current_score > 40: # Blueprint standard threshold
            self.rep_count += 1
            self.feedback = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)" # Standard feedback
        else:
            self.feedback = "Rep Failed - Form too poor"