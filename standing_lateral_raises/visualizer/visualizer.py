import cv2
import numpy as np
from collections import deque

class StandingLateralRaiseVisualizer:
    def __init__(self):
        # Color Palette
        self.COLORS = {
            'GOOD': (0, 255, 0),      # Green
            'WARNING': (0, 255, 255), # Yellow
            'ERROR': (0, 0, 255),     # Red
            'INFO': (255, 255, 0),    # Cyan
            'HUD': (255, 255, 255),   # White
            'SHADOW': (0, 0, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_DUPLEX
        
        # Path tracking for "Wing Arcs"
        self.l_wrist_path = deque(maxlen=15)
        self.r_wrist_path = deque(maxlen=15)

    def draw(self, frame, packet):
        if packet is None:
            return frame

        # Unpack Packet Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        landmarks = packet.get('coords', None)
        avg_angle = packet.get('angle', 0)

        # Determine Global Status Color
        status_color = self.COLORS['GOOD']
        if score < 70: status_color = self.COLORS['ERROR']
        elif score < 90: status_color = self.COLORS['WARNING']

        if landmarks:
            # 1. Draw "Wing Arcs" (Visualizing the path of the weights)
            self._draw_wing_arcs(frame, landmarks, faults)

            # 2. Draw "Symmetry Leveler" (The Shoulder Bar)
            self._draw_symmetry_leveler(frame, landmarks, faults)

            # 3. Draw Abduction Gauges (Split view if asymmetric)
            self._draw_angle_gauges(frame, landmarks, faults, avg_angle)

            # 4. Draw Kinetic Skeleton (Torso focus)
            self._draw_skeleton(frame, landmarks, faults)

        # 5. Render HUD and Toast Messages
        self._render_hud(frame, state, reps, score, status_color)
        if feedback:
            self._render_toast(frame, feedback, status_color)

        return frame

    def _draw_wing_arcs(self, frame, lm, faults):
        """Draws trailing paths for wrists to visualize the abduction arc."""
        h, w = frame.shape[:2]
        for i, idx in enumerate([15, 16]):  # L_Wrist, R_Wrist
            path = self.l_wrist_path if i == 0 else self.r_wrist_path
            curr_pt = (int(lm[idx].x * w), int(lm[idx].y * h))
            path.append(curr_pt)
            
            color = self.COLORS['INFO']
            if "ASYMMETRY" in faults: color = self.COLORS['WARNING']
            
            for j in range(1, len(path)):
                thickness = int(np.sqrt(15 / float(j + 1)) * 2)
                cv2.line(frame, path[j-1], path[j], color, thickness)

    def _draw_symmetry_leveler(self, frame, lm, faults):
        """Draws a horizontal line between shoulders to monitor tilt and shrugging."""
        h, w = frame.shape[:2]
        l_sh, r_sh = lm[11], lm[12]
        p1 = (int(l_sh.x * w), int(l_sh.y * h))
        p2 = (int(r_sh.x * w), int(r_sh.y * h))
        
        # Color logic for the leveler
        color = self.COLORS['HUD']
        if "SHRUGGING" in faults: color = self.COLORS['ERROR']
        elif "ASYMMETRY" in faults: color = self.COLORS['WARNING']
        
        # Draw the main leveler bar
        cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, p1, 5, color, -1)
        cv2.circle(frame, p2, 5, color, -1)

    def _draw_angle_gauges(self, frame, lm, faults, avg_angle):
        """Displays numerical angle data near the shoulders."""
        h, w = frame.shape[:2]
        mid_sh_x = int(((lm[11].x + lm[12].x) / 2) * w)
        mid_sh_y = int(((lm[11].y + lm[12].y) / 2) * h) - 30

        if "ASYMMETRY" in faults:
            # Show individual angles if asymmetry is detected
            # We would ideally pull l_angle/r_angle from the packet
            # For now, we visualize the warning state
            cv2.putText(frame, "ASYMMETRY DETECTED", (mid_sh_x - 80, mid_sh_y - 20),
                        self.FONT, 0.6, self.COLORS['WARNING'], 2)
        else:
            cv2.putText(frame, f"{avg_angle} DEG", (mid_sh_x - 40, mid_sh_y),
                        self.FONT, 0.8, self.COLORS['INFO'], 2)

    def _draw_skeleton(self, frame, lm, faults):
        """Draws the core kinetic segments: Spine, Shoulders, Arms."""
        h, w = frame.shape[:2]
        # (Start_Index, End_Index)
        connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
        
        for s_idx, e_idx in connections:
            color = self.COLORS['HUD']
            # Highlight Sway
            if "BODY_SWAY" in faults and (s_idx >= 23 or e_idx >= 23):
                color = self.COLORS['ERROR']
            
            p1 = (int(lm[s_idx].x * w), int(lm[s_idx].y * h))
            p2 = (int(lm[e_idx].x * w), int(lm[e_idx].y * h))
            cv2.line(frame, p1, p2, color, 2)

    def _render_hud(self, frame, state, reps, score, color):
        h, w = frame.shape[:2]
        # Background Bar
        cv2.rectangle(frame, (0, 0), (w, 70), self.COLORS['SHADOW'], -1)
        # Stats
        cv2.putText(frame, f"REPS: {reps}", (30, 45), self.FONT, 1, self.COLORS['HUD'], 2)
        cv2.putText(frame, f"FORM: {score}", (w//2 - 80, 45), self.FONT, 1, color, 2)
        cv2.putText(frame, state, (w - 200, 45), self.FONT, 0.8, self.COLORS['INFO'], 2)

    def _render_toast(self, frame, text, color):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1, 2)[0]
        tx = (w - text_size[0]) // 2
        # Drop shadow for readability
        cv2.putText(frame, text, (tx+2, h-48), self.FONT, 1, self.COLORS['SHADOW'], 3)
        cv2.putText(frame, text, (tx, h-50), self.FONT, 1, color, 2)