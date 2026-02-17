import cv2
import numpy as np
from collections import deque

class BarbellRowVisualizer:
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
        
        # History for Bar Path (Last 45 frames)
        self.path_history = deque(maxlen=45) 

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        Args:
            frame: Raw video frame.
            packet: Output from BarbellRowRep.process().
        """
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        torso_angle = packet.get('torso_angle', 0)
        raw_coords = packet.get('raw_coords', None)

        # 1. Determine Status Color
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            h, w = frame.shape[:2]
            
            # 2. Draw The "Hinge Protractor" & Spine
            self._draw_hinge_protractor(frame, raw_coords, torso_angle, faults, h, w)
            
            # 3. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults, h, w)
            
            # 4. Draw Bar Path Tracer
            self._draw_bar_path(frame, raw_coords, faults, h, w)

        # 5. Draw HUD (Top Bar)
        self._draw_hud(frame, state, reps, score, status_color)

        # 6. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_hinge_protractor(self, frame, landmarks, angle, faults, h, w):
        """Draws the spine, horizontal reference, and angle text."""
        # Check visibility for critical landmarks (shoulders, hips)
        l_sh = landmarks[11]; r_sh = landmarks[12]
        l_hip = landmarks[23]; r_hip = landmarks[24]
        
        if (l_sh.visibility < 0.5 or r_sh.visibility < 0.5 or 
            l_hip.visibility < 0.5 or r_hip.visibility < 0.5):
            return

        # Averages for side-agnostic drawing
        sx = (l_sh.x + r_sh.x) / 2
        sy = (l_sh.y + r_sh.y) / 2
        hx = (l_hip.x + r_hip.x) / 2
        hy = (l_hip.y + r_hip.y) / 2
        
        shoulder_pt = (int(sx * w), int(sy * h))
        hip_pt = (int(hx * w), int(hy * h))
        
        # Determine Color based on faults
        spine_color = self.COLORS['WHITE']
        if "TORSO_HEAVE" in faults:
            spine_color = self.COLORS['RED']
        elif "SHORT_PULL" in faults:
            spine_color = self.COLORS['YELLOW']
            
        # Draw Spine (The "Target Plane")
        cv2.line(frame, shoulder_pt, hip_pt, spine_color, 4, cv2.LINE_AA)
        
        # Draw Horizontal Reference Line from Hip
        # Draw it extending forward (towards the side the shoulder is on)
        direction = 1 if sx > hx else -1
        ref_pt = (int(hx * w) + (100 * direction), int(hy * h))
        cv2.line(frame, hip_pt, ref_pt, self.COLORS['CYAN'], 2, cv2.LINE_AA)
        
        # Draw Angle Text
        angle_text = f"{angle} deg"
        text_pos = (int(hx * w) + (20 * direction), int(hy * h) - 20)
        cv2.putText(frame, angle_text, text_pos, self.FONT, 0.7, spine_color, 2)

    def _draw_bar_path(self, frame, landmarks, faults, h, w):
        """Draws a trailing line behind the wrists."""
        l_wr = landmarks[15]; r_wr = landmarks[16]
        
        if l_wr.visibility < 0.5 or r_wr.visibility < 0.5:
            return

        wx = (l_wr.x + r_wr.x) / 2
        wy = (l_wr.y + r_wr.y) / 2
        
        center = (int(wx * w), int(wy * h))
        
        # Jump Cut Check: Clear history if movement is too large (e.g., user stepped away)
        if self.path_history:
            last_pt = self.path_history[-1]
            dist = np.sqrt((center[0] - last_pt[0])**2 + (center[1] - last_pt[1])**2)
            if dist > (w * 0.1): # >10% of screen width jump
                self.path_history.clear()

        self.path_history.append(center)
        
        path_color = self.COLORS['CYAN']
        if "SHORT_PULL" in faults:
            path_color = self.COLORS['YELLOW']
        elif "NO_EXTENSION" in faults:
            path_color = self.COLORS['ORANGE']

        # Convert to list for safe iteration
        points = list(self.path_history)
        for i in range(1, len(points)):
            thickness = int(np.sqrt(i + 1))
            cv2.line(frame, points[i - 1], points[i], path_color, thickness)
                     
        # Draw current bar position
        cv2.circle(frame, center, 6, self.COLORS['WHITE'], -1)

    def _draw_skeleton(self, frame, landmarks, faults, h, w):
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
        """Top Status Bar."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), 
                    self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), 
                    self.FONT, 1.0, color, 2)

        cv2.putText(frame, state, (w - 180, 40), 
                    self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        """Floating Feedback Box."""
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        
        pad = 20
        # Calculate box coordinates ensuring it's on screen
        x1 = max(0, center_x - pad)
        y1 = max(0, center_y - text_size[1] - pad)
        x2 = min(w, center_x + text_size[0] + pad)
        y2 = min(h, center_y + pad)

        cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLORS['BLACK'], -1)
        
        # Re-center text within the potentially clamped box if needed
        # (For simplicity, just draw at calculated center)
        text_x = center_x
        text_y = center_y
        
        cv2.putText(frame, text, (text_x, text_y), 
                    self.FONT, 1.2, color, 3)