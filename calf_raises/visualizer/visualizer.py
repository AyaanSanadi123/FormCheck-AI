import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        """
        Initializes the Calf Raise Visualizer with Blueprint standard colors.
        """
        self.COLOR_GOOD = (0, 255, 0)     # Green
        self.COLOR_WARNING = (0, 165, 255) # Orange
        self.COLOR_BAD = (0, 0, 255)      # Red
        self.COLOR_TEXT = (255, 255, 255) # White
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering method for the Calf Raise analysis.
        """
        # Handle null packets if user is lost
        if packet is None:
            cv2.putText(frame, "SEARCHING FOR USER...", (50, 50), 
                        self.FONT, 1, self.COLOR_WARNING, 2)
            return frame

        h, w, _ = frame.shape
        raw_coords = packet.get("raw_coords")
        faults = packet.get("faults", [])
        
        # 1. Draw Skeleton and Mechanical Levers
        if raw_coords:
            self._draw_mechanics(frame, raw_coords, faults, h, w)

        # 2. Draw HUD (Reps, Score, State)
        self._draw_hud(frame, packet, w)

        # 3. Draw Feedback Toast
        self._draw_toast(frame, packet.get("feedback", ""), h, w)

        return frame

    def _draw_mechanics(self, frame, coords, faults, h, w):
        """Draws the primary levers (Knee-Ankle-Heel) and highlights faults."""
        try:
            # Indices: Hip(24), Knee(26), Ankle(28), Heel(30), Toe(32)
            # We draw the side closest to the camera based on visibility in real-time
            active_side_idx = 26 if coords[26].visibility > coords[25].visibility else 25
            side_offset = 0 if active_side_idx == 26 else -1
            
            knee = (int(coords[26+side_offset].x * w), int(coords[26+side_offset].y * h))
            ankle = (int(coords[28+side_offset].x * w), int(coords[28+side_offset].y * h))
            heel = (int(coords[30+side_offset].x * w), int(coords[30+side_offset].y * h))
            toe = (int(coords[32+side_offset].x * w), int(coords[32+side_offset].y * h))

            # Color logic: Red if knee is bending
            leg_color = self.COLOR_BAD if "KNEE_BEND" in faults else self.COLOR_GOOD
            
            # Draw primary leg and foot levers
            cv2.line(frame, knee, ankle, leg_color, 4)
            cv2.line(frame, ankle, heel, self.COLOR_GOOD, 4)
            cv2.line(frame, ankle, toe, self.COLOR_GOOD, 2) # Foot base

            # Highlight points
            cv2.circle(frame, knee, 6, leg_color, -1)
            cv2.circle(frame, heel, 8, self.COLOR_TEXT, -1)
            
        except (IndexError, AttributeError):
            pass

    def _draw_hud(self, frame, packet, w):
        """Displays Reps, Score, and State at the top as per standards."""
        state = packet.get("state", "IDLE")
        reps = packet.get("reps", 0)
        score = packet.get("score", 100)

        # HUD Background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        cv2.putText(frame, f"REPS: {reps}", (20, 40), self.FONT, 1, self.COLOR_TEXT, 2)
        cv2.putText(frame, f"STATE: {state}", (w // 2 - 100, 40), self.FONT, 0.8, self.COLOR_WARNING, 2)
        
        score_color = self.COLOR_GOOD if score > 80 else self.COLOR_BAD
        cv2.putText(frame, f"SCORE: {score}", (w - 200, 40), self.FONT, 1, score_color, 2)

    def _draw_toast(self, frame, feedback, h, w):
        """Renders the feedback message in a conspicuous location."""
        if not feedback:
            return

        text_size = cv2.getTextSize(feedback.upper(), self.FONT, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        # Draw background toast
        cv2.rectangle(frame, (text_x - 10, h - 80), (text_x + text_size[0] + 10, h - 40), (0, 0, 0), -1)
        cv2.putText(frame, feedback.upper(), (text_x, h - 50), self.FONT, 1, self.COLOR_TEXT, 2)