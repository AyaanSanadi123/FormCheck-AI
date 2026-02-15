import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        """
        Initializes the Leg Press Visualizer with Blueprint standard colors.
        """
        self.COLOR_GOOD = (0, 255, 0)      # Green
        self.COLOR_WARNING = (0, 165, 255)  # Orange
        self.COLOR_BAD = (0, 0, 255)       # Red
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering method for Leg Press analysis.
        """
        if packet is None:
            cv2.putText(frame, "SEARCHING FOR USER...", (50, 50), 
                        self.FONT, 1, self.COLOR_WARNING, 2)
            return frame

        h, w, _ = frame.shape
        raw_coords = packet.get("raw_coords")
        faults = packet.get("faults", [])
        
        # 1. Draw Skeleton and Safety Gauges
        if raw_coords:
            self._draw_safety_skeleton(frame, raw_coords, faults, h, w)

        # 2. Draw HUD (Reps, Score, State)
        self._draw_hud(frame, packet, w)

        # 3. Draw Feedback Toast
        self._draw_toast(frame, packet.get("feedback", ""), h, w)

        return frame

    def _draw_safety_skeleton(self, frame, coords, faults, h, w):
        """Draws the leg press mechanics with specific focus on joint safety."""
        try:
            # Detect active side for raw coordinate drawing
            active_side_idx = 24 if coords[24].visibility > coords[23].visibility else 23
            side_offset = 0 if active_side_idx == 24 else -1
            
            hip = (int(coords[24+side_offset].x * w), int(coords[24+side_offset].y * h))
            knee = (int(coords[26+side_offset].x * w), int(coords[26+side_offset].y * h))
            ankle = (int(coords[28+side_offset].x * w), int(coords[28+side_offset].y * h))

            # Logic Colors
            knee_color = self.COLOR_BAD if "KNEE_LOCKOUT" in faults or "KNEE_VALGUS" in faults else self.COLOR_GOOD
            hip_color = self.COLOR_BAD if "BUTT_WINK" in faults else self.COLOR_GOOD
            
            # Draw Levers
            cv2.line(frame, hip, knee, self.COLOR_GOOD, 4)
            cv2.line(frame, knee, ankle, self.COLOR_GOOD, 4)

            # Draw Safety Pivots
            cv2.circle(frame, hip, 10, hip_color, -1)   # Monitor Hip migration
            cv2.circle(frame, knee, 12, knee_color, 2)  # Monitor Lockout/Valgus
            cv2.circle(frame, ankle, 6, self.COLOR_TEXT, -1)

            # Optional: Depth Bar (Visualizing Sled Path)
            cv2.line(frame, (w - 30, h - 50), (w - 30, 50), (100, 100, 100), 2)
            current_y = int(coords[28+side_offset].y * h)
            cv2.circle(frame, (w - 30, current_y), 8, self.COLOR_WARNING, -1)

        except (IndexError, AttributeError):
            pass

    def _draw_hud(self, frame, packet, w):
        """Displays performance metrics in the top header."""
        state = packet.get("state", "IDLE")
        reps = packet.get("reps", 0)
        score = packet.get("score", 100)

        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1, self.COLOR_TEXT, 2)
        cv2.putText(frame, f"STATE: {state}", (w // 2 - 80, 40), self.FONT, 0.8, self.COLOR_WARNING, 2)
        
        score_color = self.COLOR_GOOD if score > 80 else self.COLOR_BAD
        cv2.putText(frame, f"SCORE: {score}", (w - 200, 40), self.FONT, 1, score_color, 2)

    def _draw_toast(self, frame, feedback, h, w):
        """Renders the instructional feedback toast."""
        if not feedback: return

        text_size = cv2.getTextSize(feedback.upper(), self.FONT, 0.9, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        cv2.rectangle(frame, (text_x - 10, h - 80), (text_x + text_size[0] + 10, h - 40), (0, 0, 0), -1)
        cv2.putText(frame, feedback.upper(), (text_x, h - 50), self.FONT, 0.9, self.COLOR_TEXT, 2)