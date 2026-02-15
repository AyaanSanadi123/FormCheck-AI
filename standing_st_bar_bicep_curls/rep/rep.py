import numpy as np
import time

# Define a helper class for points, outside of the main class for efficiency
class Point:
    def __init__(self, x, y): self.x, self.y = x, y

class BicepCurlRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper
        self.scale_factor = calibration_data.get('scale_factor', 1.0) # Humerus length
        self.elbow_x_anchor = calibration_data.get('elbow_x_anchor', 0.5)
        self.neutral_spine_vector = calibration_data.get('neutral_spine_vector', [0, -1]) # [dx, dy]
        
        # Thresholds
        self.THRESH_FLEXION_PEAK = 45.0  # Degrees (Lower is more contracted)
        self.THRESH_FULL_EXTENSION = 150.0 # Degrees (Start/End)
        self.THRESH_ELBOW_DRIFT = 0.15   # Normalized X-drift for elbow
        self.THRESH_BACK_LEAN = 10.0     # Degrees of spinal deviation
        self.THRESH_DROP_SPEED = 10      # Degrees per second for uncontrolled descent
        self.THRESH_VELOCITY_GATE = 5  # Degrees/sec to detect state change
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.min_angle_achieved = 180 # Peak contraction is the minimum angle
        self.prev_angle = 180
        self.prev_time = 0
        self.angle_velocity = 0

    def process(self, landmarks, raw_landmarks=None, timestamp=None):
        """
        Main Logic Pipeline, compliant with the blueprint.
        """
        if not landmarks: return None

        # --- STEP 1: CALCULATE DT ---
        dt = 0
        if timestamp and self.prev_time:
            dt = timestamp - self.prev_time
        self.prev_time = timestamp

        # --- STEP 2: FETCH JOINTS (Normalized Profile) ---
        sh = landmarks[12] # Shoulder (Normalized)
        elb = landmarks[14] # Elbow (Normalized)
        wrist = landmarks[16] # Wrist (Normalized)
        hip = landmarks[24] # Hip (Normalized)
        
        # --- STEP 3: CALCULATE METRICS ---
        
        # Bicep Flexion Angle
        current_angle = self._calculate_joint_angle(sh, elb, wrist)
        
        # Angle Velocity (change in angle per second)
        if dt > 0:
            self.angle_velocity = (current_angle - self.prev_angle) / dt
        self.prev_angle = current_angle

        # Back Lean Angle (from current spine vector to neutral spine vector)
        current_spine_vec = np.array([sh.x - hip.x, sh.y - hip.y])
        neutral_spine_vec = np.array(self.neutral_spine_vector)
        
        curr_lean = self._calculate_vector_angle(current_spine_vec, neutral_spine_vec)

        # Elbow Drift (The "Swing" Detector)
        # Difference between current elbow X and the calibrated anchor (normalized)
        drift = abs(elb.x - self.elbow_x_anchor / self.scale_factor) if self.scale_factor else 0


        # --- STEP 4: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Stand ready for curl"
            # Trigger CONCENTRIC when arm starts curling (angle decreases, velocity is negative)
            if self.angle_velocity < -self.THRESH_VELOCITY_GATE:
                self._reset_rep(current_angle)
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Curl up!"
            self.min_angle_achieved = min(self.min_angle_achieved, current_angle)
            
            # FAULT: Elbow Swing (Forward/Backward swing)
            if drift > self.THRESH_ELBOW_DRIFT:
                self._add_fault("ELBOW_SWAY", 15, "Keep Elbows Pinned")

            # FAULT: Back Lean (Using momentum)
            if curr_lean > self.THRESH_BACK_LEAN:
                self._add_fault("BACK_LEAN", 10, "Stay Upright - No Swinging")

            # TRANSITION to TOP when arm reaches peak contraction
            if abs(self.angle_velocity) < self.THRESH_VELOCITY_GATE or current_angle < self.THRESH_FLEXION_PEAK:
                self.state = "TOP"
                if self.min_angle_achieved > self.THRESH_FLEXION_PEAK:
                    self._add_fault("PARTIAL_ROM", 15, "Squeeze Higher at Top")

        elif self.state == "TOP":
            self.feedback_buffer = "Hold & Squeeze"
            # TRANSITION to ECCENTRIC when arm starts extending (angle velocity positive)
            if self.angle_velocity > self.THRESH_VELOCITY_GATE:
                self.state = "ECCENTRIC"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Lower slowly..."
            # FAULT: Uncontrolled Descent Speed
            if self.angle_velocity > self.THRESH_DROP_SPEED:
                self._add_fault("CONTROL", 10, "Lower Slowly")

            # TRANSITION to COMPLETE when arm is fully extended
            if current_angle > self.THRESH_FULL_EXTENSION:
                self.state = "COMPLETE"

        elif self.state == "COMPLETE":
            self._finalize_rep_success()
            self.state = "IDLE"
        
        # --- STEP 5: PACKAGE OUTPUT ---
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
                "elbow_drift": drift,
                "back_lean": int(curr_lean)
            }
        }
        return packet

    # --- HELPERS ---
    def _calculate_joint_angle(self, p1, p2, p3):
        """Calculates the angle at the elbow (p2) using 2D points."""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 180.0
        
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle

    def _calculate_vector_angle(self, v1, v2):
        """Calculates the angle between two 2D vectors."""
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0) 
        angle = np.degrees(np.arccos(cosine_angle))
        
        return angle

    def _reset_rep(self, current_angle):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.min_angle_achieved = current_angle
        self.prev_angle = current_angle
        self.angle_velocity = 0

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