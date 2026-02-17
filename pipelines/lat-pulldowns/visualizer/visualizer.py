import cv2
import numpy as np

class LatPullVisualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'DARK_GRAY': (50, 50, 50)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        Args:
            frame: Raw video frame.
            packet: Output from LatPullRep.process().
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

        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords:
            h, w = frame.shape[:2]
            
            # 1. Draw The "Spirit Level" (Wrist connection)
            self._draw_spirit_level(frame, raw_coords, faults, h, w)

            # 2. Draw The "Spine Tracker" (Momentum Lean)
            self._draw_spine(frame, raw_coords, faults, h, w)
            
            # 3. Draw The Target Line (Shoulder height)
            self._draw_target_line(frame, raw_coords, h, w)
            
            # 4. Draw ROM Progress Bar
            self._draw_rom_meter(frame, raw_coords, h, w)
            
            # 5. Draw Basic Skeleton
            self._draw_skeleton(frame, raw_coords, h, w)

        # 6. Draw HUD & Toast
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_spirit_level(self, frame, landmarks, faults, h, w):
        """Draws a line between wrists. Turns red if asymmetric."""
        l_wr = landmarks[15]
        r_wr = landmarks[16]
        
        if l_wr.visibility > 0.5 and r_wr.visibility > 0.5:
            p1 = (int(l_wr.x * w), int(l_wr.y * h))
            p2 = (int(r_wr.x * w), int(r_wr.y * h))
            
            color = self.COLORS['CYAN']
            if "ASYMMETRIC_PULL" in faults:
                color = self.COLORS['RED']
                
            cv2.line(frame, p1, p2, color, 6, cv2.LINE_AA)
            cv2.circle(frame, p1, 8, color, -1)
            cv2.circle(frame, p2, 8, color, -1)

    def _draw_spine(self, frame, landmarks, faults, h, w):
        """Draws the spine. Turns red if swinging back."""
        # 11/12 Shoulders, 23/24 Hips
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        l_hip = landmarks[23]
        r_hip = landmarks[24]

        # Check visibility for all required landmarks
        if (l_sh.visibility > 0.5 and r_sh.visibility > 0.5 and
            l_hip.visibility > 0.5 and r_hip.visibility > 0.5):
            
            sx = (l_sh.x + r_sh.x) / 2
            sy = (l_sh.y + r_sh.y) / 2
            hx = (l_hip.x + r_hip.x) / 2
            hy = (l_hip.y + r_hip.y) / 2
            
            p1 = (int(sx * w), int(sy * h))
            p2 = (int(hx * w), int(hy * h))
            
            color = self.COLORS['WHITE']
            if "MOMENTUM_SWING" in faults:
                color = self.COLORS['RED']
                
            cv2.line(frame, p1, p2, color, 4, cv2.LINE_AA)

    def _draw_target_line(self, frame, landmarks, h, w):
        """Draws a horizontal dashed line at shoulder height."""
        l_sh = landmarks[11]
        r_sh = landmarks[12]

        if l_sh.visibility > 0.5 and r_sh.visibility > 0.5:
            sy = (l_sh.y + r_sh.y) / 2
            target_y = int(sy * h)
            
            # Draw dashed line across the screen
            for x in range(0, w, 30):
                cv2.line(frame, (x, target_y), (x + 15, target_y), self.COLORS['YELLOW'], 2)

    def _draw_rom_meter(self, frame, landmarks, h, w):
        """Draws a vertical progress bar based on wrist vs shoulder height."""
        # Check visibility for shoulders and wrists
        if (landmarks[11].visibility < 0.5 or landmarks[12].visibility < 0.5 or
            landmarks[15].visibility < 0.5 or landmarks[16].visibility < 0.5):
            return

        # Calculate visual ROM percentage
        # 0% = Ear/Head height (arms up), 100% = Shoulder height (arms down)
        # Using Ear midpoint (7, 8) as top anchor since Nose (0) is occluded in back view
        if landmarks[7].visibility > 0.5 and landmarks[8].visibility > 0.5:
            top_y = (landmarks[7].y + landmarks[8].y) / 2
        else:
            # Fallback to nose if ears invisible (rare but possible), or skip
            top_y = landmarks[0].y
            
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        wrist_y = (landmarks[15].y + landmarks[16].y) / 2
        
        # Guard against zero division if user leans completely horizontal
        range_y = shoulder_y - top_y
        if range_y < 0.01:
            range_y = 0.01
            
        rom_pct = (wrist_y - top_y) / range_y
        rom_pct = max(0.0, min(1.0, rom_pct)) # Clamp 0 to 1
        
        # Draw Background Bar (Right side of screen)
        bar_w = 30
        bar_h = 300
        start_x = w - 50
        start_y = h // 2 - bar_h // 2
        
        cv2.rectangle(frame, (start_x, start_y), (start_x + bar_w, start_y + bar_h), self.COLORS['DARK_GRAY'], -1)
        
        # Determine Color
        fill_color = self.COLORS['RED']
        if rom_pct > 0.85:
            fill_color = self.COLORS['GREEN']
        elif rom_pct > 0.50:
            fill_color = self.COLORS['YELLOW']
            
        # Draw Fill
        fill_h = int(bar_h * rom_pct)
        # Draw from bottom up
        cv2.rectangle(frame, (start_x, start_y + bar_h - fill_h), 
                      (start_x + bar_w, start_y + bar_h), fill_color, -1)
                      
        # Target Line on the Bar (85% mark)
        target_mark_y = start_y + bar_h - int(bar_h * 0.85)
        cv2.line(frame, (start_x - 5, target_mark_y), (start_x + bar_w + 5, target_mark_y), self.COLORS['YELLOW'], 2)

    def _draw_skeleton(self, frame, landmarks, h, w):
        """Draws the arms and torso."""
        connections = [
            (11, 13), (13, 15), # Left Arm
            (12, 14), (14, 16), # Right Arm
            (11, 23), (12, 24), # Torso Sides
            (23, 24)            # Hips
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), self.COLORS['BLACK'], -1)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        cv2.putText(frame, score_text, ((w - text_size[0]) // 2, 40), self.FONT, 1.0, color, 2)

        cv2.putText(frame, state, (w - 180, 40), self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        cx, cy = w // 2, h // 2
        
        pad = 20
        cv2.rectangle(frame, (cx - text_size[0]//2 - pad, cy - text_size[1] - pad), 
                      (cx + text_size[0]//2 + pad, cy + pad), self.COLORS['BLACK'], -1)
        
        cv2.putText(frame, text, (cx - text_size[0]//2, cy), self.FONT, 1.2, color, 3)