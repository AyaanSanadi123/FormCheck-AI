import cv2
import numpy as np

class ReversePecDeckVisualizer:
    def __init__(self):
        # Color Palette (B, G, R) - OpenCV uses BGR
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'ORANGE': (0, 165, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'CYAN': (255, 255, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        """
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)

        # 1. Determine Status Color
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['ORANGE']

        if raw_coords:
            # 2. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults)
            
        # 3. Draw HUD (Top Bar)
        self.draw_hud(frame, state, reps, score, status_color)

        # 4. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_skeleton(self, frame, landmarks, faults):
        """Draws stick figure for Upper Body."""
        h, w = frame.shape[:2]
        
        connections = [
            (11, 13), (13, 15), # Left Arm
            (12, 14), (14, 16), # Right Arm
            (11, 12), (23, 24), # Shoulders & Hips
            (11, 23), (12, 24)  # Torso Sides
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # Draw Joints
        joint_indices = [11, 12, 13, 14, 15, 16]
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                color = self.COLORS['GREEN']
                if "SHOULDER_SHRUG" in faults and (idx == 11 or idx == 12):
                    color = self.COLORS['RED']
                if "ELBOW_PIVOT" in faults and (idx == 13 or idx == 14):
                    color = self.COLORS['RED']
                
                cv2.circle(frame, (cx, cy), 5, color, -1)

    def draw_hud(self, frame, state, reps, score, color):
        """Draws the top status bar."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Reps
        cv2.putText(frame, f"REPS: {reps}", (20, 40), 
                    self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        # Score
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), 
                    self.FONT, 1.0, color, 2)

        # State
        cv2.putText(frame, state, (w - 150, 40), 
                    self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        """Draws a floating message box in the center."""
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
