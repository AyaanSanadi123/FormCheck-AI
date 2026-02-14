import cv2
import numpy as np

class HammerCurlVisualizer:
    def __init__(self):
        self.COLORS = {
            'GOOD': (0, 255, 0),      # Green
            'WARNING': (0, 255, 255), # Yellow
            'ERROR': (0, 0, 255),     # Red
            'HUD': (255, 255, 255),   # White
            'PIVOT': (255, 0, 255)    # Magenta
        }
        self.FONT = cv2.FONT_HERSHEY_DUPLEX

    def draw(self, frame, packet):
        if packet is None: return frame

        # Unpack Packet
        state = packet.get('state', "IDLE")
        reps = packet.get('reps', 0)
        score = packet.get('score', 100)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        angle = packet.get('angle', 180)
        landmarks = packet.get('coords', None)

        h, w = frame.shape[:2]
        
        if landmarks:
            # 1. The "Elbow Anchor" Box
            # Shows where the elbow SHOULD stay
            self._draw_elbow_anchor(frame, landmarks, faults)

            # 2. The Flexion Gauge
            # Visualizes the joint angle closing
            self._draw_flexion_gauge(frame, landmarks, angle)

            # 3. The Spine Plumb Line
            # Visualizes back lean (assuming a back lean fault could be added later)
            self._draw_spine_integrity(frame, landmarks, faults)

        # 4. HUD Overlays
        self._draw_hud(frame, reps, score, state, feedback)
            
        return frame

    def _draw_elbow_anchor(self, frame, lm, faults):
        """Draws a fixed target zone for the elbow."""
        h, w = frame.shape[:2]
        elb = lm[14]
        center = (int(elb.x * w), int(elb.y * h))
        
        color = self.COLORS['PIVOT']
        if "ELBOW_SWING" in faults: # Using ELBOW_SWING from HammerCurlRep
            color = self.COLORS['ERROR']
            cv2.circle(frame, center, 15, color, 2)
        
        # Draw small crosshair on the elbow pivot
        cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 20, 2)

    def _draw_flexion_gauge(self, frame, lm, angle):
        """Draws the angle text near the elbow."""
        h, w = frame.shape[:2]
        elb = (int(lm[14].x * w), int(lm[14].y * h))
        
        # Color based on ROM quality (adapt thresholds if needed for hammer curls)
        color = self.COLORS['GOOD'] if angle < 60 else self.COLORS['HUD']
        
        cv2.putText(frame, f"{angle}deg", (elb[0] + 20, elb[1] - 20), 
                    self.FONT, 0.7, color, 2)

    def _draw_spine_integrity(self, frame, lm, faults):
        """Draws a vertical reference line to highlight back lean."""
        h, w = frame.shape[:2]
        sh = (int(lm[12].x * w), int(lm[12].y * h))
        hip = (int(lm[24].x * w), int(lm[24].y * h))
        
        color = self.COLORS['HUD']
        if "BACK_LEAN" in faults: # Placeholder, assuming BACK_LEAN fault could be added to rep.py
            color = self.COLORS['ERROR']
            # Draw an arrow indicating the direction of cheat
            cv2.arrowedLine(frame, sh, (sh[0] + 40, sh[1]), color, 3)

        cv2.line(frame, sh, hip, color, 3)

    def _draw_hud(self, frame, reps, score, state, feedback):
        h, w = frame.shape[:2]
        # Top HUD Bar
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1, self.COLORS['HUD'], 2)
        cv2.putText(frame, f"SCORE: {score}", (w//2 - 50, 40), self.FONT, 1, self.COLORS['GOOD'], 2)
        
        # Bottom Feedback Toast
        if feedback:
            cv2.putText(frame, feedback.upper(), (w//2 - 150, h - 30), 
                        self.FONT, 0.8, self.COLORS['WARNING'], 2)