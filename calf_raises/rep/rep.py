import numpy as np
import time

class RepLogic:
    def __init__(self, calibration_data):
        """
        Initializes Calf Raise Rep Logic.
        Standard: Up-First (IDLE -> CONCENTRIC -> TOP -> ECCENTRIC).
        """
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.SCORE_MAX = 100 # Added for consistency
        self.current_score = self.SCORE_MAX
        self.feedback = "Ready"
        self.faults = [] # Store as list of dicts for consistency
        
        # Thresholds (Normalized Units based on Torso Scale)
        self.THRESH_CONCENTRIC_START = -0.05  # Heel begins lift
        self.THRESH_TOP_PEAK = -0.18         # Required height for valid rep
        self.THRESH_KNEE_BEND = 165.0        # Degrees (Angle < 165 = Fault)
        self.THRESH_DROP_VELO = 2.0          # Max units/sec for eccentric control
        
        # Tracking
        self.prev_heel_y = 0
        self.prev_time = None # Initialized for dt calculation (Corrected to None)

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Processes standardized landmarks to track calf raise reps and quality.
        """
        if not landmarks:
            return None

        # Handle initial prev_time setup
        if self.prev_time is None:
            self.prev_time = timestamp if timestamp is not None else time.time()
            # Return an initial packet, not a full logic run
            return {
                "state": self.state,
                "reps": self.rep_count,
                "score": self.current_score,
                "feedback": self.feedback,
                "faults": [f['code'] for f in self.faults],
                "coords": landmarks,
                "raw_coords": raw_landmarks,
                "metrics": {"heel_height": 0, "velocity": 0} # Initial values
            }

        # 1. Setup Data
        # In normalized space: Toe is (0,0), Heel is (30), Knee is (26), Hip is (24)
        heel = landmarks[30]
        knee = landmarks[26]
        hip = landmarks[24]
        
        curr_time = timestamp if timestamp else time.time()
        dt = max(curr_time - self.prev_time, 0.001) # Avoid division by zero
        velocity = (heel.y - self.prev_heel_y) / dt

        # 2. State Machine (Up-First: Concentric -> Top -> Eccentric -> Complete)
        if self.state == "IDLE":
            if heel.y < self.THRESH_CONCENTRIC_START:
                self._reset_rep()
                self.state = "CONCENTRIC"
                self.feedback = "Rise up on your toes!"

        elif self.state == "CONCENTRIC":
            # Check for Knee-Drive Cheat (Bending knees to help lift)
            knee_angle = self._calculate_angle(hip, knee, heel)
            if knee_angle < self.THRESH_KNEE_BEND:
                self._add_fault("KNEE_BEND", 15, "Keep knees straight")

            if heel.y < self.THRESH_TOP_PEAK:
                self.state = "TOP"
                self.feedback = "Squeeze at the top!"

        elif self.state == "TOP":
            # Transition to eccentric when heel starts moving down
            if velocity > 0.1:
                self.state = "ECCENTRIC"
                self.feedback = "Lower slowly"

        elif self.state == "ECCENTRIC":
            # Check for uncontrolled drop speed
            if velocity > self.THRESH_DROP_VELO:
                self._add_fault("DROP_SPEED", 10, "Control the descent")

            if heel.y > self.THRESH_CONCENTRIC_START:
                self.state = "COMPLETE" # Transition to COMPLETE state

        elif self.state == "COMPLETE": # New state for finalizing rep
            self._finalize_rep_success() # Increment rep count and set feedback
            self.state = "IDLE" # Reset to IDLE for next rep


        # Update trackers
        self.prev_heel_y = heel.y
        self.prev_time = curr_time

        # 3. Packet Structure (Blueprint Requirement 2.C.5)
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback,
            "faults": [f['code'] for f in self.faults], # Return only codes as list of strings
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"heel_height": abs(heel.y), "velocity": velocity}
        }

    def _calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)))

    def _reset_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []

    def _add_fault(self, code, penalty, msg):
        """Deducts points and logs fault. Idempotent per rep."""
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg}) # Store full dict
        self.feedback = msg # Always update feedback for the latest fault

    def _finalize_rep_success(self):
        """Finalizes the rep with nuanced feedback."""
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback = "Rep Failed - Form too poor"