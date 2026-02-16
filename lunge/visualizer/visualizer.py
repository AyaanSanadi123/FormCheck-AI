import cv2
import numpy as np

class LungeVisualizer:
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
        """Draws stick figure for Lunge."""
        h, w = frame.shape[:2]
        
        # Define relevant joint indices based on active side
        sh_idx = 12 if active_side == 'RIGHT' else 11
        f_hip_idx = 24 if active_side == 'RIGHT' else 23
        f_knee_idx = 26 if active_side == 'RIGHT' else 25
        f_ank_idx = 28 if active_side == 'RIGHT' else 27
        b_hip_idx = 23 if active_side == 'RIGHT' else 24 # Opposite hip
        b_knee_idx = 25 if active_side == 'RIGHT' else 26 # Opposite knee
        
        # Connections (Torso and Legs)
        connections = [
            (sh_idx, f_hip_idx), # Torso to front hip
            (f_hip_idx, f_knee_idx), # Front thigh
            (f_knee_idx, f_ank_idx),  # Front shin
            (b_hip_idx, b_knee_idx), # Back thigh
            (b_knee_idx, f_ank_idx) # Back shin (simplified to front ankle for visualization)
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # Draw Joints
        joint_indices = [sh_idx, f_hip_idx, f_knee_idx, f_ank_idx, b_hip_idx, b_knee_idx]
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                color = self.COLORS['GREEN']
                if "KNEE_SHEAR" in faults and idx == f_knee_idx:
                    color = self.COLORS['RED']
                elif "TORSO_LEAN" in faults and idx in [sh_idx, f_hip_idx]:
                    color = self.COLORS['RED']
                elif "KNEE_WOBBLE" in faults and idx == f_knee_idx:
                    color = self.COLORS['RED']
                elif "HIP_DRIFT" in faults and idx == f_hip_idx:
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
