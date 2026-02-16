import numpy as np
import time

class RepLogic:
    def __init__(self, calibration_data):
        """
        Initializes the Rep Logic with baselines from the Gatekeeper.
        """
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        self.knee_y_anchor = 0.0  # In normalized space, Knee is (0,0)
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = 100
        self.feedback = "Ready"
        self.faults = []
        
        # Thresholds (Normalized Units)
        self.THRESH_TOP_Y = -0.8       # Ankle must rise above this (Y is negative up)
        self.THRESH_BOTTOM_Y = -0.1    # Ankle must return below this
        self.THRESH_BUTT_LIFT = 0.15   # Max allowed knee vertical drift
        self.THRESH_DROP_VELO = 2.5    # Max units/sec during eccentric
        
        # Tracking for Velocity
        self.prev_ankle_y = 0
        self.prev_time = None

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Processes normalized landmarks to track reps and faults.
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
                "faults": list(set(self.faults)),
                "coords": landmarks,
                "raw_coords": raw_landmarks,
                "metrics": {"velocity": 0, "ankle_y": 0} # Initial values
            }

        # 1. Coordinate Setup (Using Normalized Landmarks)
        # In normalized space, Knee is at (0,0). We track the Ankle (28).
        ankle = landmarks[28]
        curr_time = timestamp if timestamp else time.time()
        dt = curr_time - self.prev_time if curr_time > self.prev_time else 0.001
        
        # 2. Velocity Calculation
        # Positive velocity = leg dropping
        velocity = (ankle.y - self.prev_ankle_y) / dt

        # 3. State Machine (Up-First: Concentric -> Top -> Eccentric)
        if self.state == "IDLE":
            if ankle.y < self.THRESH_BOTTOM_Y - 0.1:
                self._reset_rep()
                self.state = "CONCENTRIC"
                self.feedback = "Kick up!"

        elif self.state == "CONCENTRIC":
            # Check for Butt-Lift (Knee moving up away from seat)
            if abs(landmarks[26].y) > self.THRESH_BUTT_LIFT:
                self._add_fault("BUTT_LIFT", 10, "Keep your hips down")
            
            if ankle.y < self.THRESH_TOP_Y:
                self.state = "TOP"
                self.feedback = "Squeeze!"

        elif self.state == "TOP":
            if velocity > 0.1: # Starting to lower
                self.state = "ECCENTRIC"
                self.feedback = "Lower slowly"

        elif self.state == "ECCENTRIC":
            # Check for uncontrolled drop
            if velocity > self.THRESH_DROP_VELO:
                self._add_fault("CONTROL", 15, "Don't drop the weight")

            if ankle.y > self.THRESH_BOTTOM_Y:
                self.rep_count += 1
                self.state = "IDLE"
                self.feedback = "Good Rep!"

        # Update persistent trackers
        self.prev_ankle_y = ankle.y
        self.prev_time = curr_time

        # 4. Return Packet
        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback,
            "faults": list(set(self.faults)),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"velocity": velocity, "ankle_y": ankle.y}
        }

    def _reset_rep(self):
        self.current_score = 100
        self.faults = []

    def _add_fault(self, code, penalty, msg):
        if code not in self.faults:
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append(code)
        self.feedback = msg