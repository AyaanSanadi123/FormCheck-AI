import cv2
import numpy as np
from collections import deque

class HamstringCurlVisualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'GRAY': (100, 100, 100)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
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
        curl_angle = packet.get('curl_angle', 0)
        raw_coords = packet.get('raw_coords', None)

        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            h, w = frame.shape[:2]
            self._draw_skeleton(frame, raw_coords, h, w)
            self._draw_hip_guard(frame, raw_coords, faults, h, w)
            self._draw_radial_protractor(frame, raw_coords, curl_angle, faults, h, w)

        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_hip_guard(self, frame, landmarks, faults, h, w):
        """Visualizes hip stability against the machine pad."""
        l_hip = landmarks[23]; r_hip = landmarks[24]
        
        if l_hip.visibility < 0.5 or r_hip.visibility < 0.5:
            return

        hx = int(((l_hip.x + r_hip.x) / 2) * w)
        hy = int(((l_hip.y + r_hip.y) / 2) * h)
        
        color = self.COLORS['GREEN']
        if "HIP_LIFT" in faults:
            color = self.COLORS['RED']
            cv2.putText(frame, "KEEP HIPS DOWN", (hx - 60, hy - 40), self.FONT, 0.6, color, 2)
            
        # Draw a horizontal stability line at the hip
        cv2.line(frame, (hx - 40, hy), (hx + 40, hy), color, 3)

    def _draw_radial_protractor(self, frame, landmarks, angle, faults, h, w):
        """Draws the curling arc originating from the knee."""
        l_knee = landmarks[25]; r_knee = landmarks[26]
        l_ank = landmarks[27]; r_ank = landmarks[28]
        
        if (l_knee.visibility < 0.5 or r_knee.visibility < 0.5 or 
            l_ank.visibility < 0.5 or r_ank.visibility < 0.5):
            return

        kx = int(((l_knee.x + r_knee.x) / 2) * w)
        ky = int(((l_knee.y + r_knee.y) / 2) * h)
        
        ax = int(((l_ank.x + r_ank.x) / 2) * w)
        ay = int(((l_ank.y + r_ank.y) / 2) * h)

        # Update path history with current ankle position
        self.path_history.append((ax, ay))

        # 1. Draw Pivot Point (Knee)
        cv2.circle(frame, (kx, ky), 8, self.COLORS['CYAN'], -1)
        
        # 2. Draw Leg (Knee to Ankle)
        cv2.line(frame, (kx, ky), (ax, ay), self.COLORS['WHITE'], 4, cv2.LINE_AA)
        
        # 3. Draw Angle Arc (Trailing Path)
        path_color = self.COLORS['CYAN']
        if "SHORT_CURL" in faults: path_color = self.COLORS['YELLOW']
        
        points = list(self.path_history)
        for i in range(1, len(points)):
            thickness = int(np.sqrt(i + 1))
            cv2.line(frame, points[i - 1], points[i], path_color, thickness)
        
        # Display current angle near the ankle
        cv2.putText(frame, f"{angle} deg", (ax + 10, ay), self.FONT, 0.6, path_color, 2)

    def _draw_skeleton(self, frame, landmarks, h, w):
        connections = [(11, 23), (12, 24), (23, 25), (24, 26), (25, 27), (26, 28)]
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]; end = landmarks[end_idx]
            if start.visibility > 0.5 and end.visibility > 0.5:
                cv2.line(frame, (int(start.x * w), int(start.y * h)), 
                         (int(end.x * w), int(end.y * h)), self.COLORS['WHITE'], 1)

    def _draw_hud(self, frame, state, reps, score, color):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1.0, self.COLORS['WHITE'], 2)
        cv2.putText(frame, f"SCORE: {score}", (w//2 - 80, 40), self.FONT, 1.0, color, 2)
        cv2.putText(frame, state, (w - 200, 40), self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        tx, ty = (w - text_size[0]) // 2, h // 2
        cv2.rectangle(frame, (tx-10, ty-text_size[1]-10), (tx+text_size[0]+10, ty+10), (0,0,0), -1)
        cv2.putText(frame, text, (tx, ty), self.FONT, 1.2, color, 3)