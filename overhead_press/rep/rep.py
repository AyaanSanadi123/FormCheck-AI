import numpy as np
import time

# Define a helper class for points, outside of the main class for efficiency
class Point:
    def __init__(self, x, y): self.x, self.y = x, y

class OverheadPressRepCounter:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.shoulder_y = calibration_data.get('shoulder_y', 0)
        self.scale_factor = calibration_data.get('arm_length_scale', 1.0) # Using arm length as scale
        
        # Thresholds (Normalized to Arm Length)
        self.THRESH_LEAN_BACK = 20.0 # Max back lean angle allowed (Degrees)
        self.THRESH_SHALLOW_DEPTH = 0.1 # Max distance from shoulder to bar at bottom
        self.THRESH_DESCENT_SPEED = 0.8 # Normalized velocity threshold
        self.THRESH_VELOCITY_GATE = 0.1 # Normalized velocity to detect state changes
        
        # State Management
        self.state = "IDLE" 
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Physics Tracking
        self.prev_bar_y = 0
        self.velocity = 0
        self.prev_velocity = 0 # For acceleration
        self.max_bar_y = 0  # Track highest point in rep
        
        # Timers
        self.prev_time = 0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """Main Logic Pipeline, compliant with the blueprint."""
        if not landmarks:
            return None

        # --- STEP 1: EXTRACT KEY JOINTS & CALCULATE DT ---
        shoulder = self._get_midpoint(landmarks[11], landmarks[12])
        elbow = self._get_midpoint(landmarks[13], landmarks[14])
        wrist = self._get_midpoint(landmarks[15], landmarks[16])
        hip = self._get_midpoint(landmarks[23], landmarks[24])

        dt = 0
        if timestamp and self.prev_time:
            dt = timestamp - self.prev_time
        self.prev_time = timestamp
        
        # --- STEP 2: CALCULATE METRICS ---
        elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
        torso_angle = self._calculate_angle(hip, shoulder, elbow)

        curr_bar_y = wrist.y
        if dt > 0:
            self.velocity = (curr_bar_y - self.prev_bar_y) / dt
        self.prev_bar_y = curr_bar_y
        
        norm_vel = self.velocity / self.scale_factor if self.scale_factor else 0

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            if norm_vel < -self.THRESH_VELOCITY_GATE:
                self._reset_rep(curr_bar_y)
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Push!"
            if curr_bar_y < self.max_bar_y: 
                 self.max_bar_y = curr_bar_y

            if torso_angle < (90 - self.THRESH_LEAN_BACK):
                self._add_fault("LEAN_BACK", 10, "Don't lean back!")

            # Transition to TOP when movement slows at the peak
            if abs(norm_vel) < self.THRESH_VELOCITY_GATE:
                self.state = "TOP"
                if elbow_angle < 160:
                    self._add_fault("INCOMPLETE_LOCKOUT", 5, "Fully lock out!")

        elif self.state == "TOP":
            self.feedback_buffer = "Hold"
            # Transition to descent when bar starts moving down
            if norm_vel > self.THRESH_VELOCITY_GATE:
                self.state = "ECCENTRIC"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Control down..."
            if norm_vel > self.THRESH_DESCENT_SPEED:
                self._add_fault("CONTROL", 10, "Lower Slowly!")

            # Transition to COMPLETE when bar returns to shoulder height
            # Note: In normalized space, shoulder_y from calibration is the reference
            if curr_bar_y >= self.shoulder_y:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            depth_gap = abs(curr_bar_y - self.shoulder_y)
            if depth_gap > self.THRESH_SHALLOW_DEPTH:
                self._add_fault("SHALLOW_DEPTH", 5, "Bring bar to shoulders!")

            self._finalize_rep_success()
            self.state = "IDLE"

        # --- STEP 4: PACKAGE OUTPUT ---
        packet = {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer, 
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks,
            "raw_coords": raw_landmarks,
            "metrics": {
                "angle": int(elbow_angle),
                "velocity": self.velocity
            }
        }
        return packet

    # --- HELPERS ---
    def _reset_rep(self, initial_bar_y):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_bar_y = initial_bar_y

    def _add_fault(self, code, penalty, msg):
        if any(f['code'] == code for f in self.faults):
            return
        self.current_score = max(0, self.current_score - penalty)
        self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg 

    def _finalize_rep_success(self):
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback_buffer = "Rep Failed - Form too poor"

    def _get_midpoint(self, p1, p2):
        return Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)

    def _calculate_angle(self, a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
