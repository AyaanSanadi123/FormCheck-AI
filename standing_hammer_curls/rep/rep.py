import numpy as np
import time

# Define a helper class for points, outside of the main class for efficiency
class Point:
    def __init__(self, x, y): self.x, self.y = x, y

class HammerCurlRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.SCORE_MAX = 100
        
        # User Baselines (from Gatekeeper)
        self.scale_factor = calibration_data.get('scale_factor', 1.0) # Humerus length
        self.elbow_x_anchor = calibration_data.get('elbow_x_anchor', 0.5)
        
        # Thresholds
        self.THRESH_PEAK_ANGLE = 50.0  # Ideal contraction angle
        self.THRESH_FULL_EXTENSION = 150.0 # Angle for full extension
        self.THRESH_ELBOW_SWING = 0.12 # Normalized X-drift for elbow
        self.THRESH_GRIP_DRIFT = 0.08  # Max Z-variance for neutral grip
        self.THRESH_DROP_SPEED = 10    # Degrees per second for uncontrolled descent
        self.THRESH_VELOCITY_GATE = 5  # Degrees/sec to detect state change

        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.prev_angle = 180
        self.min_angle_achieved = 180 # Peak contraction is the minimum angle
        self.prev_time = 0
        self.angle_velocity = 0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Main Logic Pipeline, compliant with the blueprint.
        """
        if not landmarks:
            return None

        # --- STEP 1: CALCULATE DT ---
        dt = 0
        if timestamp and self.prev_time:
            dt = timestamp - self.prev_time
        self.prev_time = timestamp

        # --- STEP 2: FETCH JOINTS & CALCULATE METRICS ---
        sh = landmarks[12] # Shoulder (Normalized)
        elb = landmarks[14] # Elbow (Normalized)
        wrist = landmarks[16] # Wrist (Normalized)
        
        # Bicep Flexion Angle
        current_angle = self._calculate_angle(sh, elb, wrist)
        
        # Angle Velocity (change in angle per second)
        if dt > 0:
            self.angle_velocity = (current_angle - self.prev_angle) / dt
        self.prev_angle = current_angle

        # Elbow Drift (normalized)
        # Assumes active_side is RIGHT and elbow_x_anchor is for the active elbow
        drift = abs(elb.x - self.elbow_x_anchor / self.scale_factor) if self.scale_factor else 0

        # Grip Rotation (using raw landmarks if available, otherwise assume no fault)
        grip_drift = 0
        if raw_landmarks and len(raw_landmarks) > 20: # Check for thumb and pinky
            thumb = raw_landmarks[21]
            pinky = raw_landmarks[18]
            grip_drift = abs(thumb.z - pinky.z)

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Stand ready for curl"
            # Trigger CONCENTRIC when arm starts curling (angle decreases, velocity is negative)
            if self.angle_velocity < -self.THRESH_VELOCITY_GATE:
                self._reset_rep(current_angle)
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Curl up!"
            self.min_angle_achieved = min(self.min_angle_achieved, current_angle)
            
            # FAULT: Elbow Swing
            if drift > self.THRESH_ELBOW_SWING:
                self._add_fault("ELBOW_SWAY", 15, "Keep Elbows Pinned")
            
            # FAULT: Grip Rotate (if raw landmarks were available for calculation)
            if grip_drift > self.THRESH_GRIP_DRIFT:
                self._add_fault("GRIP_ROTATE", 10, "Keep Palms Facing Body")
            
            # Transition to TOP when arm reaches peak contraction (angle velocity near zero, or angle < peak)
            if abs(self.angle_velocity) < self.THRESH_VELOCITY_GATE or current_angle < self.THRESH_PEAK_ANGLE:
                self.state = "TOP"
                if self.min_angle_achieved > self.THRESH_PEAK_ANGLE:
                    self._add_fault("PARTIAL_ROM", 15, "Curl Higher!")

        elif self.state == "TOP":
            self.feedback_buffer = "Hold & Squeeze"
            # Transition to ECCENTRIC when arm starts extending (angle velocity positive)
            if self.angle_velocity > self.THRESH_VELOCITY_GATE:
                self.state = "ECCENTRIC"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Lower slowly..."
            # FAULT: Uncontrolled Descent Speed
            if self.angle_velocity > self.THRESH_DROP_SPEED:
                self._add_fault("CONTROL", 10, "Lower Slower!")

            # Transition to COMPLETE when arm is fully extended
            if current_angle > self.THRESH_FULL_EXTENSION:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
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
                "angle": int(current_angle),
                "angle_velocity": self.angle_velocity,
                "elbow_drift": drift
            }
        }
        return packet

    # --- HELPERS ---
    def _reset_rep(self, current_angle):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.min_angle_achieved = current_angle
        self.prev_angle = current_angle # Reset prev_angle for velocity calculation

    def _add_fault(self, code, penalty, msg):
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg

    def _finalize_rep_success(self):
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback_buffer = "Rep Failed - Form too poor"

    def _calculate_angle(self, p1, p2, p3):
        # p1=Shoulder, p2=Elbow (vertex), p3=Wrist
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0 # Avoid division by zero
        
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)