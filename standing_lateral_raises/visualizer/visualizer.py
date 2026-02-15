import cv2
import numpy as np
from collections import deque
import mediapipe as mp # Import for MediaPipe PoseLandmark indices

class StandingLateralRaiseVisualizer:
    def __init__(self):
        # Color Palette (B, G, R) - OpenCV uses BGR
        self.COLORS = {
            'GREEN': (0, 255, 0),      # Good/Pass
            'ORANGE': (0, 165, 255),   # Warning (Blueprint standard)
            'RED': (0, 0, 255),        # Bad/Fail
            'WHITE': (255, 255, 255),  # Text/HUD
            'BLACK': (0, 0, 0),        # Background
            'CYAN': (255, 255, 0)      # General Info (used for skeleton lines too)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        self.MP_POSE = mp.solutions.pose.PoseLandmark

        # Path tracking for "Wing Arcs" (Wrists)
        self.l_wrist_path = deque(maxlen=15)
        self.r_wrist_path = deque(maxlen=15)

    def draw(self, frame, packet):
        """
        Main rendering pipeline, compliant with blueprint.
        Args:
            frame: Raw video frame.
            packet: Output from RepLogic.process().
        """
        if packet is None:
            # If no packet, return frame or display a "Searching..." message
            h, w = frame.shape[:2]
            cv2.putText(frame, "SEARCHING...", (w // 2 - 100, h // 2), 
                        self.FONT, 1, self.COLORS['WHITE'], 2)
            return frame

        # Unpack Packet Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)
        metrics = packet.get('metrics', {}) # Get the metrics dictionary
        
        # Extract specific metrics for visualization
        avg_angle = metrics.get('avg_angle', 0)
        l_angle = metrics.get('l_angle', 0)
        r_angle = metrics.get('r_angle', 0)

        # 1. Determine Status Color (Blueprint compliant)
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['ORANGE']

        if raw_coords:
            # 2. Draw Kinetic Skeleton (Torso and Arms)
            self._draw_skeleton(frame, raw_coords, faults)

            # 3. Draw "Wing Arcs" (Visualizing the path of the weights)
            self._draw_wing_arcs(frame, raw_coords, faults)

            # 4. Draw "Symmetry Leveler" (The Shoulder Bar)
            self._draw_symmetry_leveler(frame, raw_coords, faults)

            # 5. Draw Abduction Gauges (Split view if asymmetric, or avg angle)
            self._draw_angle_gauges(frame, raw_coords, faults, avg_angle, l_angle, r_angle)

        # 6. Render HUD (Top Bar)
        self._render_hud(frame, state, reps, score, status_color)
        
        # 7. Render Toast Message (Center Screen)
        if feedback:
            self._render_toast(frame, feedback, status_color)

        return frame

    def _draw_wing_arcs(self, frame, lm, faults):
        """Draws trailing paths for wrists to visualize the abduction arc."""
        h, w = frame.shape[:2]
        for i, idx in enumerate([self.MP_POSE.LEFT_WRIST.value, self.MP_POSE.RIGHT_WRIST.value]):
            path = self.l_wrist_path if i == 0 else self.r_wrist_path
            curr_pt = (int(lm[idx].x * w), int(lm[idx].y * h))
            path.append(curr_pt)
            
            color = self.COLORS['CYAN'] # Default for path
            if "ASYMMETRY" in faults: color = self.COLORS['ORANGE'] # Warning color
            
            for j in range(1, len(path)):
                thickness = int(np.sqrt(15 / float(j + 1)) * 2)
                cv2.line(frame, path[j-1], path[j], color, thickness)

    def _draw_symmetry_leveler(self, frame, lm, faults):
        """Draws a horizontal line between shoulders to monitor tilt and shrugging."""
        h, w = frame.shape[:2]
        l_sh, r_sh = lm[self.MP_POSE.LEFT_SHOULDER.value], lm[self.MP_POSE.RIGHT_SHOULDER.value]
        p1 = (int(l_sh.x * w), int(l_sh.y * h))
        p2 = (int(r_sh.x * w), int(r_sh.y * h))
        
        # Color logic for the leveler (Blueprint compliant colors)
        color = self.COLORS['WHITE']
        if "SHRUGGING" in faults: color = self.COLORS['RED']
        elif "ASYMMETRY" in faults: color = self.COLORS['ORANGE']
        
        cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)

    def _draw_angle_gauges(self, frame, lm, faults, avg_angle, l_angle, r_angle):
        """Displays numerical angle data near the shoulders."""
        h, w = frame.shape[:2]
        mid_sh_x = int(((lm[self.MP_POSE.LEFT_SHOULDER.value].x + lm[self.MP_POSE.RIGHT_SHOULDER.value].x) / 2) * w)
        mid_sh_y = int(((lm[self.MP_POSE.LEFT_SHOULDER.value].y + lm[self.MP_POSE.RIGHT_SHOULDER.value].y) / 2) * h) - 30

        if "ASYMMETRY" in faults:
            # Show individual angles if asymmetry is detected
            l_wrist = lm[self.MP_POSE.LEFT_WRIST.value]
            r_wrist = lm[self.MP_POSE.RIGHT_WRIST.value]
            
            cv2.putText(frame, f"L: {int(l_angle)}", (int(l_wrist.x * w) - 50, int(l_wrist.y * h) - 20),
                        self.FONT, 0.6, self.COLORS['ORANGE'], 2)
            cv2.putText(frame, f"R: {int(r_angle)}", (int(r_wrist.x * w) + 10, int(r_wrist.y * h) - 20),
                        self.FONT, 0.6, self.COLORS['ORANGE'], 2)
        else:
            cv2.putText(frame, f"{int(avg_angle)} DEG", (mid_sh_x - 40, mid_sh_y),
                        self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_skeleton(self, frame, lm, faults):
        """Draws the core kinetic segments: Spine, Shoulders, Arms, compliant with blueprint."""
        h, w = frame.shape[:2]
        # Connections for full body
        connections = [
            (self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value),
            (self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value),
            (self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_ELBOW.value),
            (self.MP_POSE.LEFT_ELBOW.value, self.MP_POSE.LEFT_WRIST.value),
            (self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_ELBOW.value),
            (self.MP_POSE.RIGHT_ELBOW.value, self.MP_POSE.RIGHT_WRIST.value),
            (self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.LEFT_HIP.value),
            (self.MP_POSE.RIGHT_SHOULDER.value, self.MP_POSE.RIGHT_HIP.value),
            (self.MP_POSE.LEFT_HIP.value, self.MP_POSE.LEFT_KNEE.value),
            (self.MP_POSE.LEFT_KNEE.value, self.MP_POSE.LEFT_ANKLE.value),
            (self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.RIGHT_KNEE.value),
            (self.MP_POSE.RIGHT_KNEE.value, self.MP_POSE.RIGHT_ANKLE.value),
        ]
        
        for s_idx, e_idx in connections:
            start = lm[s_idx]
            end = lm[e_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                color = self.COLORS['WHITE'] # Default skeleton color
                # Highlight Sway fault on torso
                if "BODY_SWAY" in faults and (s_idx in [self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value] or \
                                             e_idx in [self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value, self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value]):
                    color = self.COLORS['RED']
                
                cv2.line(frame, p1, p2, color, 2)

        # Draw Joints and apply fault-based coloring
        # Only draw critical joints for clarity
        joint_indices = [
            self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value,
            self.MP_POSE.LEFT_ELBOW.value, self.MP_POSE.RIGHT_ELBOW.value,
            self.MP_POSE.LEFT_WRIST.value, self.MP_POSE.RIGHT_WRIST.value,
            self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value,
            self.MP_POSE.LEFT_KNEE.value, self.MP_POSE.RIGHT_KNEE.value,
            self.MP_POSE.LEFT_ANKLE.value, self.MP_POSE.RIGHT_ANKLE.value,
        ]
        
        for idx in joint_indices:
            lm_point = lm[idx]
            if lm_point.visibility > 0.5:
                cx, cy = int(lm_point.x * w), int(lm_point.y * h)
                
                color = self.COLORS['GREEN'] # Default joint color
                
                # Fault-specific coloring
                if "SHRUGGING" in faults and (idx == self.MP_POSE.LEFT_SHOULDER.value or idx == self.MP_POSE.RIGHT_SHOULDER.value):
                    color = self.COLORS['RED']
                elif "ASYMMETRY" in faults and (idx == self.MP_POSE.LEFT_WRIST.value or idx == self.MP_POSE.RIGHT_WRIST.value or \
                                               idx == self.MP_POSE.LEFT_ELBOW.value or idx == self.MP_POSE.RIGHT_ELBOW.value):
                    color = self.COLORS['ORANGE'] # Warning for asymmetry
                
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 8, self.COLORS['WHITE'], 1) # Outline


    def _render_hud(self, frame, state, reps, score, color):
        """Draws the top status bar, blueprint compliant."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Reps (Left)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), 
                    self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        # Score (Center) - Changed FORM to SCORE
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), 
                    self.FONT, 1.0, color, 2)

        # State (Right)
        state_text_size = cv2.getTextSize(state, self.FONT, 0.8, 2)[0]
        cv2.putText(frame, state, (w - state_text_size[0] - 20, 40), # 20px padding from right
                    self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _render_toast(self, frame, text, color):
        """Draws a floating message box in the center, blueprint compliant."""
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        
        pad = 20
        cv2.rectangle(frame, 
                      (center_x - pad, center_y - text_size[1] - pad), 
                      (center_x + text_size[0] + pad, center_y + pad), 
                      self.COLORS['BLACK'], -1)
        
        cv2.putText(frame, text, (center_x, center_y), 
                    self.FONT, 1.2, color, 3)