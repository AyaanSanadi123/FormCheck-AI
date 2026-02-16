import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Standardized Colors from Blueprint
        self.COLOR_GOOD = (0, 255, 0)
        self.COLOR_WARNING = (0, 165, 255)
        self.COLOR_BAD = (0, 0, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering method. Modified frame is returned.
        Handles null packets gracefully as per Blueprint Section 2.D.
        """
        if packet is None:
            cv2.putText(frame, "SEARCHING FOR USER...", (50, 50), 
                        self.FONT, 1, self.COLOR_WARNING, 2)
            return frame

        h, w, _ = frame.shape
        raw_coords = packet.get("raw_coords")
        faults = packet.get("faults", [])
        active_side = packet.get("active_side", "RIGHT") # Default to RIGHT if not found
        
        # 1. Draw Skeleton (Using Raw Coordinates)
        if raw_coords:
            self._draw_skeleton(frame, raw_coords, faults, h, w, active_side)

        # 2. Draw HUD (Reps, Score, State)
        self._draw_hud(frame, packet, w)

        # 3. Draw Feedback Toast
        self._draw_toast(frame, packet.get("feedback", ""), h, w)

        return frame

    def _draw_skeleton(self, frame, coords, faults, h, w, active_side):
        """Draws the primary mechanical levers for the leg extension."""
        # Joints: Hip(24/23), Knee(26/25), Ankle(28/27)
        hip_idx = 24 if active_side == "RIGHT" else 23
        knee_idx = 26 if active_side == "RIGHT" else 25
        ankle_idx = 28 if active_side == "RIGHT" else 27

        try:
            hip = (int(coords[hip_idx].x * w), int(coords[hip_idx].y * h))
            knee = (int(coords[knee_idx].x * w), int(coords[knee_idx].y * h))
            ankle = (int(coords[ankle_idx].x * w), int(coords[ankle_idx].y * h))

            # Color logic for faults
            knee_color = self.COLOR_BAD if "BUTT_LIFT" in faults else self.COLOR_GOOD
            
            # Draw Thigh and Shin Levers
            cv2.line(frame, hip, knee, self.COLOR_GOOD, 4)
            cv2.line(frame, knee, ankle, self.COLOR_GOOD, 4)

            # Highlight Knee Pivot
            cv2.circle(frame, knee, 8, knee_color, -1)
            cv2.circle(frame, ankle, 6, self.COLOR_TEXT, -1)
        except (IndexError, AttributeError):
            pass

    def _draw_hud(self, frame, packet, w):
        """Draws the performance metrics at the top of the frame."""
        state = packet.get("state", "IDLE")
        reps = packet.get("reps", 0)
        score = packet.get("score", 100)

        # Background Header
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1, self.COLOR_TEXT, 2)
        cv2.putText(frame, f"STATE: {state}", (w // 2 - 100, 40), self.FONT, 0.8, self.COLOR_WARNING, 2)
        
        score_color = self.COLOR_GOOD if score > 80 else self.COLOR_BAD
        cv2.putText(frame, f"SCORE: {score}", (w - 200, 40), self.FONT, 1, score_color, 2)

    def _draw_toast(self, frame, feedback, h, w):
        """Displays short UI messages in a conspicuous location."""
        if not feedback:
            return

        text_size = cv2.getTextSize(feedback.upper(), self.FONT, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        # Draw background for text
        cv2.rectangle(frame, (text_x - 10, h - 80), (text_x + text_size[0] + 10, h - 40), (0, 0, 0), -1)
        cv2.putText(frame, feedback.upper(), (text_x, h - 50), self.FONT, 1, self.COLOR_TEXT, 2)