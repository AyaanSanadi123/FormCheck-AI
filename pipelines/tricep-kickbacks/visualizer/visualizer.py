import cv2
import numpy as np

class Visualizer:
    """
    Visualizer for Cable Tricep Kickback.
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
            'BLACK': (0, 0, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _get_val(lm, attr):
        if isinstance(lm, dict):
            return lm.get(attr)
        return getattr(lm, attr)

    def draw(self, frame, packet):
        h, w = frame.shape[:2]
        
        if packet is None:
            return self._draw_searching(frame)

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
        torso_angle = metrics.get('torso_angle', 0)

        # Status color for HUD (Overall Rep Quality)
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['ORANGE']

        # Feedback color (Contextual)
        toast_color = self.COLORS['WHITE']
        if faults:
            toast_color = self.COLORS['RED'] if any(f in ["PENDULUM_SWING", "TORSO_BOB"] for f in faults) else self.COLORS['ORANGE']

        if raw_coords and len(raw_coords) >= 33:
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
                self._draw_torso_line(frame, sh, hip, torso_angle, faults, h, w)
                self._draw_arm_lever(frame, sh, el, wr, elbow_angle, faults, h, w)
                self._draw_anchor_ring(frame, el, faults, h, w)
            else:
                cv2.putText(frame, "WAITING FOR CLEAR VIEW...", (w//2 - 120, h//2 + 100), 
                            self.FONT, 0.7, self.COLORS['ORANGE'], 2, cv2.LINE_AA)

        # Draw UI Overlays
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, toast_color)

        return frame

    def _draw_searching(self, frame):
        """Draws an overlay when no user is detected or pipeline is reset."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//2 - 200, h//2 - 30), (w//2 + 200, h//2 + 30), self.COLORS['BLACK'], -1)
        cv2.putText(frame, "SEARCHING FOR LIFTER...", (w//2 - 160, h//2 + 10), 
                    self.FONT, 0.8, self.COLORS['WHITE'], 2, cv2.LINE_AA)
        return frame

    def _draw_torso_line(self, frame, sh, hip, angle, faults, h, w):
        """Draws the back angle line and flashes red if 'Torso Bob' cheating is detected."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        hip_pt = (int(self._get_val(hip, 'x') * w), int(self._get_val(hip, 'y') * h))
        
        torso_color = self.COLORS['GREEN']
        thickness = 4
        
        if "TORSO_BOB" in faults:
            torso_color = self.COLORS['RED']
            thickness = 8
            # Add a warning text near the center of the torso
            mid_x, mid_y = (sh_pt[0] + hip_pt[0]) // 2, (sh_pt[1] + hip_pt[1]) // 2
            cv2.putText(frame, "BACK HEAVING!", (mid_x - 60, mid_y - 20), 
                        self.FONT, 0.6, self.COLORS['RED'], 2, cv2.LINE_AA)
            
        cv2.line(frame, hip_pt, sh_pt, torso_color, thickness, cv2.LINE_AA)
        cv2.putText(frame, f"{angle} deg", (hip_pt[0] - 20, hip_pt[1] + 30), 
                    self.FONT, 0.5, torso_color, 2, cv2.LINE_AA)

    def _draw_arm_lever(self, frame, sh, el, wr, angle, faults, h, w):
        """Draws the active kicking arm."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        wr_pt = (int(self._get_val(wr, 'x') * w), int(self._get_val(wr, 'y') * h))
        
        upper_arm_color = self.COLORS['WHITE']
        forearm_color = self.COLORS['WHITE']
        
        if "SHORT_LOCKOUT" in faults:
            forearm_color = self.COLORS['YELLOW']
        if "PENDULUM_SWING" in faults:
            upper_arm_color = self.COLORS['RED']
            
        cv2.line(frame, sh_pt, el_pt, upper_arm_color, 4, cv2.LINE_AA)
        cv2.line(frame, el_pt, wr_pt, forearm_color, 4, cv2.LINE_AA)
        
        # Draw Angle Text near the wrist
        cv2.putText(frame, f"{angle} deg", (wr_pt[0] + 15, wr_pt[1] - 15), 
                    self.FONT, 0.6, forearm_color, 2, cv2.LINE_AA)

    def _draw_anchor_ring(self, frame, el, faults, h, w):
        """Draws the pivot ring on the elbow to monitor pendulum swing cheating."""
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        
        if "PENDULUM_SWING" in faults:
            cv2.circle(frame, el_pt, 20, self.COLORS['RED'], 4, cv2.LINE_AA)
            cv2.circle(frame, el_pt, 30, self.COLORS['RED'], 2, cv2.LINE_AA)
            cv2.putText(frame, "ANCHOR BROKEN", (el_pt[0] - 60, el_pt[1] - 40), 
                        self.FONT, 0.5, self.COLORS['RED'], 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, el_pt, 12, self.COLORS['CYAN'], 2, cv2.LINE_AA)
            cv2.circle(frame, el_pt, 4, self.COLORS['GREEN'], -1, cv2.LINE_AA)

    def _draw_hud(self, frame, state, reps, score, color):
        """Standardized HUD."""
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
        """Standardized floating message with slight transparency."""
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        
        pad = 20
        x1, y1 = max(0, center_x - pad), max(0, center_y - text_size[1] - pad)
        x2, y2 = min(w, center_x + text_size[0] + pad), min(h, center_y + pad)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, text, (center_x, center_y), self.FONT, 1.2, color, 3, cv2.LINE_AA)