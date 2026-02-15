import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        """
        Initializes the Preacher Curl Visualizer with Blueprint standard colors.
        """
        self.COLOR_GOOD = (0, 255, 0)      # Green
        self.COLOR_WARNING = (0, 165, 255)  # Orange
        self.COLOR_BAD = (0, 0, 255)       # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering method for Preacher Curl analysis.
        """
        if packet is None:
            cv2.putText(frame, "SEARCHING FOR USER...", (50, 50), 
                        self.FONT, 1, self.COLOR_WARNING, 2)
            return frame

        h, w, _ = frame.shape
        raw_coords = packet.get("raw_coords")
        faults = packet.get("faults", [])
        
        # 1. Draw Skeleton and Pad-Contact Indicators
        if raw_coords:
            self._draw_arm_mechanics(frame, raw_coords, faults, h, w)

        # 2. Draw HUD (Reps, Score, State)
        self._draw_hud(frame, packet, w)

        # 3. Draw Feedback Toast
        self._draw_toast(frame, packet.get("feedback", ""), h, w)

        return frame

    def _draw_arm_mechanics(self, frame, coords, faults, h, w):
        """Draws the biceps lever and highlights elbow contact with the pad."""
        try:
            # Detect active side based on shoulder visibility
            active_side_idx = 12 if coords[12].visibility > coords[11].visibility else 11
            side_offset = 0 if active_side_idx == 12 else -1
            
            shoulder = (int(coords[12+side_offset].x * w), int(coords[12+side_offset].y * h))
            elbow = (int(coords[14+side_offset].x * w), int(coords[14+side_offset].y * h))
            wrist = (int(coords[16+side_offset].x * w), int(coords[16+side_offset].y * h))

            # Logic Colors
            pad_color = self.COLOR_BAD if "ELBOW_LIFT" in faults else self.COLOR_GOOD
            rom_color = self.COLOR_WARNING if "SHORT_ROM" in faults else self.COLOR_GOOD
            
            # Draw Humerus (Upper Arm) - Represents the segment against the pad
            cv2.line(frame, shoulder, elbow, pad_color, 6)
            
            # Draw Forearm (The moving lever)
            cv2.line(frame, elbow, wrist, rom_color, 4)

            # Draw "Pad Baseline" - A simple line to represent the preacher bench
            # Drawn at a 45-degree angle from the elbow for visual context
            cv2.line(frame, (elbow[0]-40, elbow[1]+20), (elbow[0]+60, elbow[1]-80), (100, 100, 100), 2)

            # Highlight Joint Pivots
            cv2.circle(frame, elbow, 10, pad_color, -1)   # The anchor point
            cv2.circle(frame, wrist, 8, self.COLOR_TEXT, -1)
            
        except (IndexError, AttributeError):
            pass

    def _draw_hud(self, frame, packet, w):
        """Displays Reps, Score, and State at the top as per standards."""
        state = packet.get("state", "IDLE")
        reps = packet.get("reps", 0)
        score = packet.get("score", 100)
        angle = packet.get("metrics", {}).get("angle", 180)

        # Header Bar
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1, self.COLOR_TEXT, 2)
        
        score_color = self.COLOR_GOOD if score > 80 else self.COLOR_BAD
        cv2.putText(frame, f"SCORE: {score}", (w - 200, 40), self.FONT, 1, score_color, 2)

    def _draw_toast(self, frame, feedback, h, w):
        """Renders the feedback message at the bottom center."""
        if not feedback: return

        text_size = cv2.getTextSize(feedback.upper(), self.FONT, 0.9, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 # Center of screen
        
        pad = 20
        cv2.rectangle(frame, 
                      (text_x - pad, text_y - text_size[1] - pad), 
                      (text_x + text_size[0] + pad, text_y + pad), 
                      (0,0,0), -1)
        cv2.putText(frame, feedback.upper(), (text_x, text_y), self.FONT, 0.9, self.COLOR_TEXT, 2)