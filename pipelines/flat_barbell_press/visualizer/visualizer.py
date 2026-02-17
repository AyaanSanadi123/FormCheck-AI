import cv2
import numpy as np
from collections import deque

class BenchVisualizer:
    def __init__(self):
        # Color Palette (B, G, R) - OpenCV uses BGR
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE_TRANSPARENT': (255, 0, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        # Visual History for Bar Path (Stores last 20 frames of Wrist coords)
        self.path_history = deque(maxlen=20) 

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        Args:
            frame: Raw video frame.
            packet: Output from BenchPressRep.process().
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
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            # 2. Draw The "Target Box" (Shoulder Alignment)
            self._draw_target_box(frame, raw_coords)

            # 3. Draw The "Elbow Flare" Triangle
            self._draw_flare_triangle(frame, raw_coords, faults)

            # 4. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults)
            
            # 5. Draw Bar Path Tracer
            self._draw_bar_path(frame, raw_coords, faults)

        # 6. Draw HUD (Top Bar)
        self._draw_hud(frame, state, reps, score, status_color)

        # 7. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_target_box(self, frame, landmarks):
        """Draws a target zone over the shoulders (Where bar should end)."""
        h, w = frame.shape[:2]
        
        # Get Shoulder Coordinates
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        
        if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
            # Calculate box around shoulders
            sx = int((l_sh.x + r_sh.x) / 2 * w)
            sy = int((l_sh.y + r_sh.y) / 2 * h)
            
            # Draw semi-transparent box
            overlay = frame.copy()
            box_size = 40
            cv2.rectangle(overlay, (sx - box_size, sy - box_size), 
                          (sx + box_size, sy + box_size), self.COLORS['BLUE_TRANSPARENT'], -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Label
            cv2.putText(frame, "TARGET", (sx - 30, sy - 50), 
                        self.FONT, 0.5, self.COLORS['CYAN'], 1, cv2.LINE_AA)

    def _draw_bar_path(self, frame, landmarks, faults):
        """Draws a trailing line behind the wrists to show the arc."""
        h, w = frame.shape[:2]
        
        # Calculate Bar Center (Avg of Wrists)
        l_wrist = landmarks[15]
        r_wrist = landmarks[16]
        
        if l_wrist.visibility > 0.5 and r_wrist.visibility > 0.5:
            cx = int((l_wrist.x + r_wrist.x) / 2 * w)
            cy = int((l_wrist.y + r_wrist.y) / 2 * h)
            
            self.path_history.append((cx, cy))
            
            # Determine Color (Red if "BAD_PATH" fault is active)
            path_color = self.COLORS['CYAN']
            if "BAD_PATH" in faults or "BOUNCE" in faults:
                path_color = self.COLORS['RED']

            # Draw the line
            for i in range(1, len(self.path_history)):
                thickness = int(np.sqrt(i + 1)) # Thicker at the head
                cv2.line(frame, self.path_history[i - 1], self.path_history[i], 
                         path_color, thickness)

    def _draw_flare_triangle(self, frame, landmarks, faults):
        """Draws a triangle (Shoulder-Elbow-Hip) to visualize flare."""
        h, w = frame.shape[:2]
        
        # Indices: 11=L_Sh, 13=L_Elb, 23=L_Hip (Left Side)
        # We draw on the side closest to camera (simplification: draw both for now)
        sides = [(11, 13, 23), (12, 14, 24)]
        
        for sh_idx, elb_idx, hip_idx in sides:
            sh = landmarks[sh_idx]
            elb = landmarks[elb_idx]
            hip = landmarks[hip_idx]
            
            if (sh.visibility > 0.5 and elb.visibility > 0.5 and hip.visibility > 0.5):
                pts = np.array([
                    [int(sh.x * w), int(sh.y * h)],
                    [int(elb.x * w), int(elb.y * h)],
                    [int(hip.x * w), int(hip.y * h)]
                ], np.int32)
                
                # Color Logic: Red if FLARING, otherwise Green Outline
                if "ELBOW_FLARE" in faults:
                    # Fill Red
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], self.COLORS['RED'])
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                else:
                    # Green Outline (Safe)
                    cv2.polylines(frame, [pts], True, self.COLORS['GREEN'], 1)

    def _draw_skeleton(self, frame, landmarks, faults):
        """Draws stick figure for Upper Body."""
        h, w = frame.shape[:2]
        
        # Bench Press Skeleton: Arms + Shoulders + Hips (Torso Box)
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
        joint_indices = [11, 12, 13, 14, 15, 16, 23, 24]
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Default Green
                color = self.COLORS['GREEN']
                
                # Context-Aware Fault Coloring
                if "GLUTE_BRIDGE" in faults and (idx == 23 or idx == 24):
                    color = self.COLORS['RED'] # Hips Red
                elif "ASYMMETRY" in faults and (idx == 15 or idx == 16):
                    color = self.COLORS['RED'] # Wrists Red
                
                cv2.circle(frame, (cx, cy), 5, color, -1)

    def _draw_hud(self, frame, state, reps, score, color):
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