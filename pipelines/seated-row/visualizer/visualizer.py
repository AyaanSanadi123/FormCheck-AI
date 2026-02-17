import cv2
import numpy as np
from collections import deque

class SeatedRowVisualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'ORANGE': (0, 165, 255),
            'GRAY': (100, 100, 100)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        # History for Bar Path (Last 45 frames)
        self.path_history = deque(maxlen=45) 

    def draw(self, frame, packet):
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        torso_angle = packet.get('torso_angle', 90)
        raw_coords = packet.get('raw_coords', None)

        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            h, w = frame.shape[:2]
            
            # Draw layers from back to front
            self._draw_skeleton(frame, raw_coords, h, w)
            self._draw_metronome(frame, raw_coords, torso_angle, faults, h, w)
            self._draw_target_line(frame, raw_coords, faults, h, w)
            self._draw_bar_path(frame, raw_coords, faults, h, w)

        # Draw UI Overlays
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_metronome(self, frame, landmarks, angle, faults, h, w):
        """Draws the vertical reference line and actual spine."""
        l_sh = landmarks[11]; r_sh = landmarks[12]
        l_hip = landmarks[23]; r_hip = landmarks[24]
        l_wr = landmarks[15]; r_wr = landmarks[16]
        
        if (l_sh.visibility < 0.5 or l_hip.visibility < 0.5):
            return

        # Averages for side-agnostic drawing
        sx = (l_sh.x + r_sh.x) / 2
        sy = (l_sh.y + r_sh.y) / 2
        hx = (l_hip.x + r_hip.x) / 2
        hy = (l_hip.y + r_hip.y) / 2
        wx = (l_wr.x + r_wr.x) / 2
        
        shoulder_pt = (int(sx * w), int(sy * h))
        hip_pt = (int(hx * w), int(hy * h))
        
        # 1. Draw Vertical Reference Line (The Metronome Anchor)
        # Extends straight up from the hip
        ref_top = (int(hx * w), int(hy * h) - 150)
        cv2.line(frame, hip_pt, ref_top, self.COLORS['GRAY'], 2, cv2.LINE_AA)
        
        # 2. Draw Actual Spine
        spine_color = self.COLORS['WHITE']
        if "MOMENTUM_SWING" in faults:
            spine_color = self.COLORS['RED']
        elif "SHRUGGING" in faults:
            spine_color = self.COLORS['ORANGE']
            
        cv2.line(frame, shoulder_pt, hip_pt, spine_color, 4, cv2.LINE_AA)
        
        # 3. Draw Angle Text
        # Position it "behind" the user (opposite to reach direction)
        # If wrists are to the right of hips (wx > hx), user reaches right. Text goes left (-1).
        direction = -1 if wx > hx else 1
        text_pos = (int(hx * w) + (60 * direction), int(hy * h) - 40)
        cv2.putText(frame, f"{angle} deg", text_pos, self.FONT, 0.6, spine_color, 2)

    def _draw_target_line(self, frame, landmarks, faults, h, w):
        """Draws a vertical dashed line representing the torso boundary."""
        l_hip = landmarks[23]; r_hip = landmarks[24]
        l_wr = landmarks[15]; r_wr = landmarks[16]
        
        if l_hip.visibility < 0.5:
            return

        hx = (l_hip.x + r_hip.x) / 2
        hy = (l_hip.y + r_hip.y) / 2
        wx = (l_wr.x + r_wr.x) / 2
        
        # Determine "Front" direction (towards wrists)
        direction = 1 if wx > hx else -1
        
        # Target line is slightly in front of the hips (where the stomach is)
        target_x = int(hx * w) + (40 * direction)
        start_y = int(hy * h) - 100
        end_y = int(hy * h) + 20
        
        color = self.COLORS['CYAN']
        if "SHORT_PULL" in faults:
            color = self.COLORS['YELLOW']
            
        # Draw dashed line
        for y in range(start_y, end_y, 20):
            cv2.line(frame, (target_x, y), (target_x, y + 10), color, 2)

    def _draw_bar_path(self, frame, landmarks, faults, h, w):
        """Draws a trailing line behind the wrists."""
        l_wr = landmarks[15]; r_wr = landmarks[16]
        
        if l_wr.visibility < 0.5 and r_wr.visibility < 0.5:
            return

        wx = (l_wr.x + r_wr.x) / 2
        wy = (l_wr.y + r_wr.y) / 2
        
        center = (int(wx * w), int(wy * h))
        
        # Jump Cut Check
        if self.path_history:
            last_pt = self.path_history[-1]
            dist = np.sqrt((center[0] - last_pt[0])**2 + (center[1] - last_pt[1])**2)
            if dist > (w * 0.1): # >10% of screen width jump
                self.path_history.clear()

        self.path_history.append(center)
        
        path_color = self.COLORS['CYAN']
        if "SHORT_PULL" in faults:
            path_color = self.COLORS['YELLOW']
        elif "NO_STRETCH" in faults:
            path_color = self.COLORS['ORANGE']

        points = list(self.path_history)
        for i in range(1, len(points)):
            thickness = int(np.sqrt(i + 1))
            cv2.line(frame, points[i - 1], points[i], path_color, thickness)
                     
        cv2.circle(frame, center, 6, self.COLORS['WHITE'], -1)

    def _draw_skeleton(self, frame, landmarks, h, w):
        """Draws legs and arms."""
        connections = [
            (11, 13), (13, 15), # Arms
            (12, 14), (14, 16),
            (23, 25), (25, 27), # Legs
            (24, 26), (26, 28)
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 1, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), self.FONT, 1.0, color, 2)

        cv2.putText(frame, state, (w - 180, 40), self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        
        pad = 20
        x1 = max(0, center_x - pad)
        y1 = max(0, center_y - text_size[1] - pad)
        x2 = min(w, center_x + text_size[0] + pad)
        y2 = min(h, center_y + pad)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS['BLACK'], -1)
        cv2.putText(frame, text, (center_x, center_y), self.FONT, 1.2, color, 3)