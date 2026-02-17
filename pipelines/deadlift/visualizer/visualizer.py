import cv2
import numpy as np
from collections import deque

class DeadliftVisualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'ORANGE': (0, 165, 255)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        # History for Bar Path (Last 30 frames)
        self.path_history = deque(maxlen=30) 

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        Args:
            frame: Raw video frame.
            packet: Output from DeadliftRep.process().
        """
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None) # We need RAW coords for drawing

        # 1. Determine Status Color
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            h, w = frame.shape[:2]
            
            # 2. Draw The "Plumb Line" (Reference from Mid-Foot)
            self._draw_reference_line(frame, raw_coords, h, w)

            # 3. Draw The "Spine Stick" (Back Alignment)
            self._draw_spine_line(frame, raw_coords, faults, h, w)
            
            # 4. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults, h, w)
            
            # 5. Draw Bar Path Tracer
            self._draw_bar_path(frame, raw_coords, faults, h, w)

        # 6. Draw HUD (Top Bar)
        self._draw_hud(frame, state, reps, score, status_color)

        # 7. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_reference_line(self, frame, landmarks, h, w):
        """Draws a vertical line from the ankle upwards (Ideal Path)."""
        # 27=L_Ankle, 28=R_Ankle
        l_ank = landmarks[27]
        r_ank = landmarks[28]
        
        if l_ank.visibility > 0.5 and r_ank.visibility > 0.5:
            # Calculate Mid-Foot X
            mx = int((l_ank.x + r_ank.x) / 2 * w)
            my = int((l_ank.y + r_ank.y) / 2 * h) # Ankle height
            
            # Draw vertical line going UP from ankle
            # Use dashed line effect (points)
            for y in range(my, 0, -20):
                cv2.circle(frame, (mx, y), 2, self.COLORS['WHITE'], -1)

    def _draw_spine_line(self, frame, landmarks, faults, h, w):
        """Draws a line connecting Shoulders to Hips to visualize back angle."""
        # 11/12 Shoulders, 23/24 Hips
        # Use average to get center spine
        sx = (landmarks[11].x + landmarks[12].x) / 2
        sy = (landmarks[11].y + landmarks[12].y) / 2
        hx = (landmarks[23].x + landmarks[24].x) / 2
        hy = (landmarks[23].y + landmarks[24].y) / 2
        
        p1 = (int(sx * w), int(sy * h))
        p2 = (int(hx * w), int(hy * h))
        
        # Color Logic: Red if Stripper Pull or Cat Back
        color = self.COLORS['GREEN']
        if "STRIPPER_PULL" in faults or "CAT_BACK" in faults:
            color = self.COLORS['RED']
        elif "OVER_EXTEND" in faults:
            color = self.COLORS['ORANGE']
            
        cv2.line(frame, p1, p2, color, 4, cv2.LINE_AA)

    def _draw_bar_path(self, frame, landmarks, faults, h, w):
        """Draws a trailing line behind the wrists."""
        # 15/16 Wrists
        wx = (landmarks[15].x + landmarks[16].x) / 2
        wy = (landmarks[15].y + landmarks[16].y) / 2
        
        center = (int(wx * w), int(wy * h))
        
        # Determine color for THIS point
        current_color = self.COLORS['CYAN']
        if "BAR_DRIFT" in faults:
            current_color = self.COLORS['RED']
            
        self.path_history.append((center, current_color))
        
        for i in range(1, len(self.path_history)):
            thickness = int(np.sqrt(i + 1))
            # Use the color of the *destination* point for the segment
            pt_prev, _ = self.path_history[i - 1]
            pt_curr, color = self.path_history[i]
            
            cv2.line(frame, pt_prev, pt_curr, color, thickness)

    def _draw_skeleton(self, frame, landmarks, faults, h, w):
        """Draws basic stick figure."""
        # Standard connections for Deadlift
        connections = [
            (11, 13), (13, 15), # Arms
            (12, 14), (14, 16),
            (23, 25), (25, 27), # Legs
            (24, 26), (26, 28),
            (11, 23), (12, 24)  # Torso Sides
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 1, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
        """Top Status Bar."""
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
        cv2.putText(frame, state, (w - 180, 40), 
                    self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        """Floating Feedback Box."""
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