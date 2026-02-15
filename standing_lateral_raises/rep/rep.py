import numpy as np
import time

class StandingLateralRaiseRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper (Anatomical Proportions)
        self.scale_factor = calibration_data.get('scale_factor', 1.0) # Torso length
        self.arm_length_scale = calibration_data.get('arm_length', 1.0) # Secondary scale for some thresholds
        self.neutral_hip_x = calibration_data.get('neutral_hip_x', 0.5)
        
        # Thresholds
        self.THRESH_ABDUCTION = 80.0   # Target peak (degrees)
        self.THRESH_MIN_START_ANGLE = 20.0 # Min angle for rep start
        self.THRESH_SHRUG = 0.10       # 10% reduction in torso length = shrug
        self.THRESH_SWAY = 0.08        # Horizontal hip drift (normalized)
        self.THRESH_ASYMMETRY = 15.0   # Max degree difference L vs R
        self.THRESH_DESCENT_SPEED = 0.5 # Normalized velocity threshold for controlled descent (y-axis)
        self.THRESH_VELOCITY_GATE = 0.1 # Normalized velocity threshold for state changes
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.max_angle = 0
        self.prev_wrist_y = 0
        self.wrist_velocity = 0 # Vertical velocity of wrists
        self.prev_time = 0

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
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_wrist, r_wrist = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        
        # Calculate abduction relative to the spine vector
        l_angle = self._get_abduction(l_hip, l_sh, l_wrist)
        r_angle = self._get_abduction(r_hip, r_sh, r_wrist)
        avg_angle = (l_angle + r_angle) / 2

        # PROPORTIONAL SHRUG DETECTION
        curr_torso_l = np.linalg.norm([l_sh.x - l_hip.x, l_sh.y - l_hip.y])
        shrug_ratio = curr_torso_l / self.scale_factor if self.scale_factor else 1.0

        # TORSO SWAY (MOMENTUM)
        curr_hip_x = (l_hip.x + r_hip.x) / 2
        sway_dist = abs(curr_hip_x - self.neutral_hip_x) / self.scale_factor if self.scale_factor else 0

        # VERTICAL WRIST VELOCITY
        curr_wrist_y = (l_wrist.y + r_wrist.y) / 2
        if dt > 0:
            self.wrist_velocity = (curr_wrist_y - self.prev_wrist_y) / dt
        self.prev_wrist_y = curr_wrist_y

        # --- STEP 3: STATE MACHINE & FAULT DETECTION ---
        
        if self.state == "IDLE":
            self.feedback_buffer = "Ready"
            # Trigger CONCENTRIC when arms start moving up and past minimum angle
            if avg_angle > self.THRESH_MIN_START_ANGLE and self.wrist_velocity < -self.THRESH_VELOCITY_GATE:
                self._start_rep()
                self.state = "CONCENTRIC"

        elif self.state == "CONCENTRIC":
            self.feedback_buffer = "Raise Arms!"
            self.max_angle = max(self.max_angle, avg_angle)
            
            # FAULT: Shrugging
            if shrug_ratio < (1.0 - self.THRESH_SHRUG):
                self._add_fault("SHRUGGING", 15, "Keep Shoulders Down")

            # FAULT: Sway (Using momentum)
            if sway_dist > self.THRESH_SWAY:
                self._add_fault("BODY_SWAY", 10, "No Body Sway")

            # FAULT: Asymmetry
            if abs(l_angle - r_angle) > self.THRESH_ASYMMETRY:
                self._add_fault("ASYMMETRY", 10, "Raise Arms Evenly")

            # Transition to TOP when vertical velocity near zero at peak
            if abs(self.wrist_velocity) < self.THRESH_VELOCITY_GATE:
                self.state = "TOP"
                if self.max_angle < self.THRESH_ABDUCTION:
                    self._add_fault("SHALLOW_ROM", 15, "Reach Shoulder Height")

        elif self.state == "TOP":
            self.feedback_buffer = "Hold"
            # Transition to ECCENTRIC when arms start moving down
            if self.wrist_velocity > self.THRESH_VELOCITY_GATE:
                self.state = "ECCENTRIC"

        elif self.state == "ECCENTRIC":
            self.feedback_buffer = "Lower slowly..."
            # FAULT: Uncontrolled Descent Speed
            if self.wrist_velocity > self.THRESH_DESCENT_SPEED:
                self._add_fault("CONTROL", 10, "Lower Slower!")

            # Transition to COMPLETE when arms are back at starting angle
            if avg_angle < self.THRESH_MIN_START_ANGLE:
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
                "l_angle": int(l_angle),
                "r_angle": int(r_angle),
                "avg_angle": int(avg_angle),
                "wrist_velocity": self.wrist_velocity,
                "shrug_ratio": shrug_ratio,
                "sway_dist": sway_dist
            }
        }
        return packet

    # --- HELPERS ---

    def _get_abduction(self, hip, sh, wrist):
        """
        Calculates angle between Spine Vector (Hip->Sh) and Arm Vector (Sh->Wrist).
        Uses Dot Product for 3D-perspective-safe angular measurement.
        """
        # Vector A: Spine (Points down to hip)
        spine_vec = np.array([hip.x - sh.x, hip.y - sh.y, hip.z - sh.z])
        # Vector B: Arm (Points out to wrist)
        arm_vec = np.array([wrist.x - sh.x, wrist.y - sh.y, wrist.z - sh.z])
        
        unit_spine = spine_vec / np.linalg.norm(spine_vec)
        unit_arm = arm_vec / np.linalg.norm(arm_vec)
        
        dot_product = np.dot(unit_spine, unit_arm)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # In this vector setup, 180 deg is arms at sides, 90 deg is T-pose
        # We return (180 - angle) so that 0 = sides and 90 = T-pose
        return 180 - np.degrees(angle_rad)

    def _start_rep(self):
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.max_angle = 0

    def _add_fault(self, code, penalty, msg):
        if not any(f['code'] == code for f in self.faults):
            self.current_score = max(0, self.current_score - penalty)
            self.faults.append({"code": code, "msg": msg})
        self.feedback_buffer = msg

    def _finalize_rep_success(self):
        """Finalizes the rep with nuanced feedback."""
        if self.current_score > 40:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!" if self.current_score > 80 else "Rep Counted (Watch Form)"
        else:
            self.feedback_buffer = "Rep Failed - Form too poor"