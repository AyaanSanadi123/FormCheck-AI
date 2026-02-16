import cv2
import numpy as np

class PlankVisualizer:
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
        Main rendering pipeline for Plank Isometric analysis.
        """
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0) # Reps is hold_time in seconds
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)
        active_side = packet.get('active_side', 'RIGHT') # Default to right if not provided

        # 1. Determine Status Color
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['ORANGE']

        if raw_coords:
            # 2. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults, active_side)
            
        # 3. Draw HUD (Top Bar)
        self.draw_hud(frame, state, reps, score, status_color)

        # 4. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_skeleton(self, frame, landmarks, faults, active_side):
        """Draws stick figure for Plank."""
        h, w = frame.shape[:2]
        
        # Define relevant joint indices based on active side
        ear_idx = 8 if active_side == 'RIGHT' else 7
        sh_idx = 12 if active_side == 'RIGHT' else 11
        el_idx = 14 if active_side == 'RIGHT' else 13
        hip_idx = 24 if active_side == 'RIGHT' else 23
        ank_idx = 28 if active_side == 'RIGHT' else 27

        # Connections (Body segments)
        connections = [
            (ear_idx, sh_idx), # Neck
            (sh_idx, el_idx),  # Upper arm
            (el_idx, sh_idx),  # Forearm (to draw a thick line for elbow)
            (sh_idx, hip_idx), # Torso upper
            (hip_idx, ank_idx)  # Torso lower
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                
                # Special drawing for elbow to make it look like a forearm plank
                if (start_idx == el_idx and end_idx == sh_idx) or (start_idx == sh_idx and end_idx == el_idx):
                    cv2.line(frame, p1, p2, self.COLORS['WHITE'], 8, cv2.LINE_AA) # Thicker line for forearm
                else:
                    cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # Draw Joints
        joint_indices = [ear_idx, sh_idx, el_idx, hip_idx, ank_idx]
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                color = self.COLORS['GREEN']
                if "HIP_SAG" in faults and idx == hip_idx:
                    color = self.COLORS['RED']
                elif "HIP_PIKE" in faults and idx == hip_idx:
                    color = self.COLORS['RED']
                elif "NECK_DIVE" in faults and idx == ear_idx:
                    color = self.COLORS['RED']
                elif "STABILITY_LOSS" in faults and idx in [hip_idx, ank_idx]:
                    color = self.COLORS['RED']
                
                cv2.circle(frame, (cx, cy), 5, color, -1)

    def draw_hud(self, frame, state, reps, score, color):
        """Draws the top status bar."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Time Held (Reps as seconds)
        cv2.putText(frame, f"TIME: {reps}s", (20, 40), 
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
