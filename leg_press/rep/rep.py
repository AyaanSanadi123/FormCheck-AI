import numpy as np
import time

class RepLogic:
    def __init__(self, calibration_data):
        """
        Initializes Leg Press Rep Logic.
        Standard: Down-First (IDLE -> ECCENTRIC -> BOTTOM -> CONCENTRIC -> COMPLETE).
        """
        self.scale_factor = calibration_data.get('scale_factor', 1.0)
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.SCORE_MAX = 100
        self.current_score = self.SCORE_MAX
        self.feedback = "Ready"
        self.faults = [] # Store as list of dicts for internal tracking
        
        # Thresholds (Normalized Units relative to Hip-Origin)
        # In normalized space, Hip is (0,0). Ankle Y increases as sled lowers.
        self.THRESH_ECCENTRIC_START = 0.15  # Sled moves away from full extension
        self.THRESH_BOTTOM_PEAK = 0.65      # Minimum depth for a "full" rep
        self.THRESH_KNEE_LOCK = 175.0       # Degrees (Angle > 175 = Dangerous lockout)
        self.THRESH_HIP_MIGRATION = 0.12    # Vertical Y-drift of Hip (Butt-wink)
        
        # Tracking for Physics
        self.prev_ankle_y = 0
        self.prev_time = None

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Processes standardized landmarks to track leg press form.
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
                "metrics": {"depth": 0, "knee_angle": 0} # Initial values
            }

        # 1. Setup Data
        # Coordinates: Hip(24), Knee(26), Ankle(28)
        # Note: In normalized space, Hip is (0,0).
        hip = landmarks[24]
        knee = landmarks[26]
        ankle = landmarks[28]
        
        curr_time = timestamp if timestamp else time.time()
        dt = max(curr_time - self.prev_time, 0.001)
        # Positive velocity = lowering the weight (increasing Y distance from Hip)
        velocity = (ankle.y - self.prev_ankle_y) / dt

        # 2. Safety Check: Knee Lockout (Continuous Monitoring)
        knee_angle = self._calculate_angle(hip, knee, ankle)
        if knee_angle > self.THRESH_KNEE_LOCK:
            self._add_fault("KNEE_LOCKOUT", 25, "Avoid locking knees fully")

        # 3. State Machine (Down-First)
        if self.state == "IDLE":
            if ankle.y > self.THRESH_ECCENTRIC_START and velocity > 0.1:
                self._reset_rep()
                self.state = "ECCENTRIC"
                self.feedback = "Lower slowly..."

        elif self.state == "ECCENTRIC":
            # Check for Pelvic Stability (Hip lifting off seat)
            # Since Hip is origin, any Y-drift indicates migration
            if abs(hip.y) > self.THRESH_HIP_MIGRATION:
                self._add_fault("BUTT_WINK", 15, "Keep lower back against seat")
            
            if ankle.y > self.THRESH_BOTTOM_PEAK:
                self.state = "BOTTOM"
                self.feedback = "Drive up!"

        elif self.state == "BOTTOM":
            # Transition when user starts pushing back
            if velocity < -0.1:
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            # Check for Knee Valgus (Simplified X-plane check)
            if abs(knee.x - ankle.x) > 0.15: # Knees significantly inside ankle line
                self._add_fault("KNEE_VALGUS", 10, "Keep knees in line with feet")

            if ankle.y < self.THRESH_ECCENTRIC_START:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finalize_rep_success()
            self.state = "IDLE"

        # Update persistent trackers
        self.prev_ankle_y = ankle.y
        self.prev_time = curr_time

        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback,
            "faults": [f['code'] for f in self.faults],
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {"depth": ankle.y, "knee_angle": knee_angle}
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
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg})
        self.feedback = msg

    def _finalize_rep_success(self):
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback = "Rep Failed - Form too poor"