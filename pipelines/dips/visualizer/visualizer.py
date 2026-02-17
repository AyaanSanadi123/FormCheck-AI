import cv2
import numpy as np

class Visualizer:
    """
    Visualizer for Parallel Bar Dips.
    Complies with PIPELINE_BLUEPRINT.md standards.
    """
    def __init__(self):
        # Blueprint Compliant Color Palette
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'ORANGE': (0, 165, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE': (255, 100, 100) # For the depth gauge
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def draw(self, frame, packet):
        if packet is None:
            return frame

        # Unpack Blueprint Standard Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)
        
        # Unpack Specific Metrics
        metrics = packet.get('metrics', {})
        active_side = metrics.get('active_side', "RIGHT")
        elbow_angle = metrics.get('elbow_angle', 0)

        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords and len(raw_coords) >= 33:
            h, w = frame.shape[:2]
            
            # Map indices based on the active side
            if active_side == "LEFT":
                idx_sh, idx_hip = 11, 23
                idx_el, idx_wr = 13, 15
            else:
                idx_sh, idx_hip = 12, 24
                idx_el, idx_wr = 14, 16

            sh = raw_coords[idx_sh]
            hip = raw_coords[idx_hip]
            el = raw_coords[idx_el]
            wr = raw_coords[idx_wr]

            # Visibility check
            sh_vis = self._get_val(sh, 'visibility') or 0
            hip_vis = self._get_val(hip, 'visibility') or 0
            el_vis = self._get_val(el, 'visibility') or 0
            wr_vis = self._get_val(wr, 'visibility') or 0

            # Render overlays if joints are visible
            if sh_vis > 0.5 and hip_vis > 0.5 and el_vis > 0.5 and wr_vis > 0.5:
                self._draw_torso_line(frame, sh, hip, faults, h, w)
                self._draw_arm_lever(frame, sh, el, wr, elbow_angle, faults, h, w)
                self._draw_depth_gauge(frame, sh, el, state, faults, h, w)

        # Draw UI Overlays
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_torso_line(self, frame, sh, hip, faults, h, w):
        """Draws the body line and flashes red if 'Pendulum Swing' is detected."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        hip_pt = (int(self._get_val(hip, 'x') * w), int(self._get_val(hip, 'y') * h))
        
        torso_color = self.COLORS['CYAN']
        thickness = 4
        
        if "PENDULUM_SWING" in faults:
            torso_color = self.COLORS['RED']
            thickness = 8
            # Warning text
            mid_x, mid_y = (sh_pt[0] + hip_pt[0]) // 2, (sh_pt[1] + hip_pt[1]) // 2
            cv2.putText(frame, "STOP SWINGING!", (mid_x - 70, mid_y), 
                        self.FONT, 0.6, self.COLORS['RED'], 2, cv2.LINE_AA)
            
        cv2.line(frame, hip_pt, sh_pt, torso_color, thickness, cv2.LINE_AA)

    def _draw_arm_lever(self, frame, sh, el, wr, angle, faults, h, w):
        """Draws the arm and flags lockout faults."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        wr_pt = (int(self._get_val(wr, 'x') * w), int(self._get_val(wr, 'y') * h))
        
        arm_color = self.COLORS['WHITE']
        if "SHORT_LOCKOUT" in faults:
            arm_color = self.COLORS['YELLOW']
            
        cv2.line(frame, sh_pt, el_pt, arm_color, 4, cv2.LINE_AA)
        cv2.line(frame, el_pt, wr_pt, arm_color, 4, cv2.LINE_AA)
        
        # Highlight joints
        cv2.circle(frame, sh_pt, 6, arm_color, -1, cv2.LINE_AA)
        cv2.circle(frame, el_pt, 6, arm_color, -1, cv2.LINE_AA)
        cv2.circle(frame, wr_pt, 8, self.COLORS['ORANGE'], -1, cv2.LINE_AA) # Highlight the fixed anchor
        
        # Display elbow angle
        cv2.putText(frame, f"{angle} deg", (el_pt[0] + 15, el_pt[1]), 
                    self.FONT, 0.5, arm_color, 2, cv2.LINE_AA)

    def _draw_depth_gauge(self, frame, sh, el, state, faults, h, w):
        """Draws a horizontal line from the elbow. Turns green when shoulder crosses it."""
        sh_y = int(self._get_val(sh, 'y') * h)
        el_x = int(self._get_val(el, 'x') * w)
        el_y = int(self._get_val(el, 'y') * h)
        
        # Draw a horizontal reference line spanning 100 pixels across the elbow
        line_start = (el_x - 80, el_y)
        line_end = (el_x + 80, el_y)
        
        line_color = self.COLORS['BLUE']
        thickness = 2
        
        # In OpenCV, higher Y value means lower on the screen.
        # If the shoulder drops below the elbow, sh_y becomes greater than el_y.
        is_deep_enough = sh_y >= el_y - 2 # 2 pixel buffer for stability
        
        if "SHALLOW_REP" in faults:
            line_color = self.COLORS['YELLOW']
            thickness = 4
            cv2.putText(frame, "GO LOWER!", (line_end[0] + 10, el_y + 5), 
                        self.FONT, 0.6, self.COLORS['YELLOW'], 2, cv2.LINE_AA)
        elif is_deep_enough:
            line_color = self.COLORS['GREEN']
            thickness = 4
            cv2.putText(frame, "GOOD DEPTH!", (line_end[0] + 10, el_y + 5), 
                        self.FONT, 0.6, self.COLORS['GREEN'], 2, cv2.LINE_AA)
            cv2.line(frame, line_start, line_end, line_color, thickness, cv2.LINE_AA)
        else:
            line_color = self.COLORS['BLUE']
            thickness = 2
            for i in range(line_start[0], line_end[0], 15):
                cv2.line(frame, (i, el_y), (i+10, el_y), line_color, thickness, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
        """Standardized HUD per pipeline blueprint."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1.0, self.COLORS['WHITE'], 2, cv2.LINE_AA)
        
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), self.FONT, 1.0, color, 2, cv2.LINE_AA)

        cv2.putText(frame, state, (w - 200, 40), self.FONT, 0.8, self.COLORS['CYAN'], 2, cv2.LINE_AA)

    def _draw_toast(self, frame, text, color):
        """Standardized floating message per pipeline blueprint."""
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
        cv2.putText(frame, text, (center_x, center_y), self.FONT, 1.2, color, 3, cv2.LINE_AA)