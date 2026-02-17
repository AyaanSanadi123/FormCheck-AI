import cv2
import numpy as np

class Visualizer:
    """
    Visualizer for Incline Barbell/Dumbbell Press.
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
        if packet is None:
            h, w = frame.shape[:2]
            cv2.putText(frame, "SEARCHING FOR LIFTER...", (w // 2 - 180, h // 2), 
                        self.FONT, 0.9, self.COLORS['CYAN'], 2, cv2.LINE_AA)
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
            
            # 1. Draw Full Skeleton first (Background layer)
            self._draw_full_skeleton(frame, raw_coords, h, w)

            # Map indices based on the active side for specialized overlays
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

            # 2. Render Specialized Overlays
            if sh_vis > 0.5 and hip_vis > 0.5 and el_vis > 0.5 and wr_vis > 0.5:
                self._draw_torso_line(frame, sh, hip, faults, h, w)
                self._draw_arm_plumb_line(frame, sh, el, wr, elbow_angle, faults, h, w)

        # Draw UI Overlays
        self._draw_hud(frame, state, reps, score, status_color)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_full_skeleton(self, frame, landmarks, h, w):
        """Draws a dimmed full body skeleton for context."""
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24), # Torso
            (11, 13), (13, 15), (12, 14), (14, 16), # Arms
            (23, 25), (25, 27), (24, 26), (26, 28)  # Legs
        ]
        
        for start_idx, end_idx in connections:
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            if (self._get_val(start_lm, 'visibility') or 0) > 0.5 and \
               (self._get_val(end_lm, 'visibility') or 0) > 0.5:
                
                pt1 = (int(self._get_val(start_lm, 'x') * w), int(self._get_val(start_lm, 'y') * h))
                pt2 = (int(self._get_val(end_lm, 'x') * w), int(self._get_val(end_lm, 'y') * h))
                cv2.line(frame, pt1, pt2, (100, 100, 100), 1, cv2.LINE_AA) # Dim gray

    def _draw_torso_line(self, frame, sh, hip, faults, h, w):
        """Draws the back angle line and flashes red if 'Hip Bridge' cheating is detected."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        hip_pt = (int(self._get_val(hip, 'x') * w), int(self._get_val(hip, 'y') * h))
        
        torso_color = self.COLORS['CYAN']
        thickness = 4
        
        if "HIP_BRIDGE" in faults:
            torso_color = self.COLORS['RED']
            thickness = 8
            # Warning text
            mid_x, mid_y = (sh_pt[0] + hip_pt[0]) // 2, (sh_pt[1] + hip_pt[1]) // 2
            cv2.putText(frame, "HIPS DOWN!", (mid_x - 60, mid_y), 
                        self.FONT, 0.6, self.COLORS['RED'], 2, cv2.LINE_AA)
            
        cv2.line(frame, hip_pt, sh_pt, torso_color, thickness, cv2.LINE_AA)

    def _draw_arm_plumb_line(self, frame, sh, el, wr, angle, faults, h, w):
        """Draws the active arm and a vertical reference line to check joint stacking."""
        sh_pt = (int(self._get_val(sh, 'x') * w), int(self._get_val(sh, 'y') * h))
        el_pt = (int(self._get_val(el, 'x') * w), int(self._get_val(el, 'y') * h))
        wr_pt = (int(self._get_val(wr, 'x') * w), int(self._get_val(wr, 'y') * h))
        
        arm_color = self.COLORS['WHITE']
        if "SHALLOW_REP" in faults or "SHORT_LOCKOUT" in faults:
            arm_color = self.COLORS['YELLOW']
        elif "UNSTACKED_JOINTS" in faults:
            arm_color = self.COLORS['RED']
            
        # Draw Upper Arm and Forearm
        cv2.line(frame, sh_pt, el_pt, arm_color, 4, cv2.LINE_AA)
        cv2.line(frame, el_pt, wr_pt, arm_color, 4, cv2.LINE_AA)
        
        # Highlight joints
        cv2.circle(frame, sh_pt, 6, arm_color, -1, cv2.LINE_AA)
        cv2.circle(frame, el_pt, 6, arm_color, -1, cv2.LINE_AA)
        cv2.circle(frame, wr_pt, 6, arm_color, -1, cv2.LINE_AA)

        # Draw the Vertical Plumb Line (Ideal Forearm Path)
        # Extend a dashed or thin line straight up from the elbow
        plumb_top = (el_pt[0], el_pt[1] - int(h * 0.2)) 
        cv2.line(frame, el_pt, plumb_top, self.COLORS['GREEN'], 2, cv2.LINE_AA)
        
        # Display elbow angle
        cv2.putText(frame, f"{angle} deg", (el_pt[0] + 20, el_pt[1]), 
                    self.FONT, 0.6, arm_color, 2, cv2.LINE_AA)

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