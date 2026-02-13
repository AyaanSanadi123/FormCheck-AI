import numpy as np
import time

class StandingLateralRaiseRep:
    def __init__(self, calibration_data):
        # --- CONFIGURATION ---
        self.FPS = 30
        self.SCORE_MAX = 100
        
        # Baselines from Gatekeeper (Anatomical Proportions)
        self.torso_baseline = calibration_data.get('torso_baseline', 1.0)
        self.arm_length = calibration_data.get('arm_length', 1.0)
        self.neutral_hip_x = calibration_data.get('neutral_hip_x', 0.5)
        
        # Thresholds
        self.THRESH_ABDUCTION = 80.0   # Target peak (degrees)
        self.THRESH_SHRUG = 0.10       # 10% reduction in torso length = shrug
        self.THRESH_SWAY = 0.08        # Horizontal hip drift
        self.THRESH_ASYMMETRY = 15.0   # Max degree difference L vs R
        
        # State Management
        self.state = "IDLE"
        self.rep_count = 0
        self.current_score = self.SCORE_MAX
        self.faults = []
        self.feedback_buffer = "Ready"
        
        # Tracking
        self.max_angle = 0
        self.prev_wrist_y = 0
        self.velocity = 0

    def process(self, landmarks, raw_landmarks=None):
        if not landmarks:
            return None

        # 1. FETCH JOINTS (Normalized)
        l_sh, r_sh = landmarks[11], landmarks[12]
        l_wrist, r_wrist = landmarks[15], landmarks[16]
        l_hip, r_hip = landmarks[23], landmarks[24]
        
        # 2. REFINED VECTOR CALCULATIONS
        # Calculate abduction relative to the spine vector, not screen vertical
        l_angle = self._get_abduction(l_hip, l_sh, l_wrist)
        r_angle = self._get_abduction(r_hip, r_sh, r_wrist)
        avg_angle = (l_angle + r_angle) / 2

        # 3. PROPORTIONAL SHRUG DETECTION
        # Measure current distance between shoulder and hip
        curr_torso_l = np.linalg.norm([l_sh.x - l_hip.x, l_sh.y - l_hip.y])
        shrug_ratio = curr_torso_l / self.torso_baseline

        # 4. TORSO SWAY (MOMENTUM)
        curr_hip_x = (l_hip.x + r_hip.x) / 2
        sway_dist = abs(curr_hip_x - self.neutral_hip_x)

        # 5. VELOCITY TRACKING
        curr_wrist_y = (l_wrist.y + r_wrist.y) / 2
        if self.prev_wrist_y != 0:
            self.velocity = (curr_wrist_y - self.prev_wrist_y) * self.FPS
        self.prev_wrist_y = curr_wrist_y

        # --- STATE MACHINE ---
        
        if self.state == "IDLE":
            if avg_angle > 20 and self.velocity < -0.1:
                self._start_rep()
                self.state = "ASCENDING"

        elif self.state == "ASCENDING":
            self.max_angle = max(self.max_angle, avg_angle)
            
            # FAULT: Shrugging (Torso "shortens" as shoulders rise to ears)
            if shrug_ratio < (1.0 - self.THRESH_SHRUG):
                self._add_fault("SHRUGGING", 15, "Keep Shoulders Down")

            # FAULT: Sway (Using momentum)
            if sway_dist > self.THRESH_SWAY:
                self._add_fault("BODY_SWAY", 10, "Stop the Body Swing")

            # FAULT: Asymmetry
            if abs(l_angle - r_angle) > self.THRESH_ASYMMETRY:
                self._add_fault("ASYMMETRY", 10, "Raise Arms Evenly")

            if self.velocity > 0.05 and avg_angle > 45:
                self.state = "DESCENDING"
                if self.max_angle < self.THRESH_ABDUCTION:
                    self._add_fault("SHALLOW", 15, "Reach Shoulder Height")

        elif self.state == "DESCENDING":
            if avg_angle < 25:
                self._finish_rep()
                self.state = "IDLE"

        return {
            "state": self.state,
            "reps": self.rep_count,
            "score": self.current_score,
            "feedback": self.feedback_buffer,
            "angle": int(avg_angle),
            "faults": list(set([f['code'] for f in self.faults])),
            "coords": landmarks
        }

    # --- MATH HELPERS ---

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

    def _finish_rep(self):
        if self.current_score > 60:
            self.rep_count += 1
            self.feedback_buffer = "Good Rep!"
        else:
            self.feedback_buffer = "Rep Failed - Watch Form"