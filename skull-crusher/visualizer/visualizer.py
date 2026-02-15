import cv2
import numpy as np
from collections import deque

class SkullCrusherVisualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'CYAN': (255, 255, 0),
            'YELLOW': (0, 255, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.path_history = deque(maxlen=150) 

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def draw(self, frame, packet):
        if packet is None:
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)
        
        # Unpack Metrics
        metrics = packet.get('metrics', {})
        elbow_angle = metrics.get('elbow_angle', 0)
        active_side = metrics.get('active_side', "RIGHT")

        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['YELLOW']

        if raw_coords and len(raw_coords) >= 33:
            h, w = frame.shape[:2]
            
            # Determine which indices to draw based on the active side
            if active_side == "LEFT":
                idx_sh, idx_el, idx_wr = 11, 13, 15
            else:
                idx_sh, idx_el, idx_wr = 12, 14, 16

            sh = raw_coords[idx_sh]
            el = raw_coords[idx_el]
            wr = raw_coords[idx_wr]

            # Use helper to safe-access attributes
            sh_vis = self._get_val(sh, 'visibility')
            el_vis = self._get_val(el, 'visibility')
            wr_vis = self._get_val(wr, 'visibility')

            # Draw if the active arm is reasonably visible
            if sh_vis > 0.5 and el_vis > 0.5 and wr_vis > 0.5:
                self._draw_lever_arm(frame, sh, el, wr, elbow_angle, faults, h, w)
                self._draw_anchor_ring(frame, el, faults, h, w)
                self._draw_wrist_path(frame, wr, faults, h, w)

        # Draw UI Overlays
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_lever_arm(self, frame, sh, el, wr, angle, faults, h, w):
        """Draws the mechanical lever of the arm and the interior angle."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        wr_pt = (int(self._get_val(wr, 'x') * w), int(self._get_val(wr, 'y') * h))
        
        arm_color = self.COLORS['WHITE']
        if "SHALLOW_DROP" in faults or "SHORT_LOCKOUT" in faults:
            arm_color = self.COLORS['YELLOW']
        elif "SHOULDER_SWING" in faults:
            arm_color = self.COLORS['RED']
            
        # Draw Upper Arm and Forearm
        cv2.line(frame, sh_pt, el_pt, arm_color, 4, cv2.LINE_AA)
        cv2.line(frame, el_pt, wr_pt, arm_color, 4, cv2.LINE_AA)
        
        # Draw Angle Text near the elbow (offset slightly)
        cv2.putText(frame, f"{angle} deg", (el_pt[0] + 20, el_pt[1] - 20), 
                    self.FONT, 0.6, arm_color, 2, cv2.LINE_AA)

    def _draw_anchor_ring(self, frame, el, faults, h, w):
        """Draws the pivot ring on the elbow to monitor lat pullover cheating."""
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        
        if "SHOULDER_SWING" in faults:
            cv2.circle(frame, el_pt, 15, self.COLORS['RED'], 4, cv2.LINE_AA)
            cv2.circle(frame, el_pt, 25, self.COLORS['RED'], 2, cv2.LINE_AA)
            cv2.putText(frame, "ANCHOR BROKEN", (el_pt[0] - 60, el_pt[1] - 35), 
                        self.FONT, 0.5, self.COLORS['RED'], 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, el_pt, 12, self.COLORS['CYAN'], 2, cv2.LINE_AA)
            cv2.circle(frame, el_pt, 4, self.COLORS['GREEN'], -1, cv2.LINE_AA)

    def _draw_wrist_path(self, frame, wr, faults, h, w):
        """Draws a trailing line behind the active wrist."""
        center = (int(self._get_val(wr, 'x') * w), int(self._get_val(wr, 'y') * h))
        
        if self.path_history:
            last_pt = self.path_history[-1]
            dist = np.sqrt((center[0] - last_pt[0])**2 + (center[1] - last_pt[1])**2)
            if dist > (w * 0.1): 
                self.path_history.clear()

        self.path_history.append(center)
        
        path_color = self.COLORS['CYAN']
        if "SHALLOW_DROP" in faults or "SHORT_LOCKOUT" in faults:
            path_color = self.COLORS['YELLOW']

        points = list(self.path_history)
        for i in range(1, len(points)):
            thickness = min(int(np.sqrt(i + 1)), 10)
            cv2.line(frame, points[i - 1], points[i], path_color, thickness, cv2.LINE_AA)
                     
        cv2.circle(frame, center, 6, self.COLORS['WHITE'], -1, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
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