import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Color Palette (B, G, R)
        self.COLORS = {
            'GREEN': (0, 255, 0),
            'RED': (0, 0, 255),
            'ORANGE': (0, 165, 255),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'BLUE': (255, 0, 0)
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def draw(self, frame, packet):
        """
        Main rendering pipeline.
        Args:
            frame: The raw video frame from the camera.
            packet: The dictionary output from SquatRep.process().
        Returns:
            The processed frame with overlays.
        """
        if packet is None: 
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        angle = packet.get('angle', 0)
        faults = packet.get('faults', [])
        
        # 1. Determine Global Status Color
        status_color = self.COLORS['GREEN']
        if score < 70:
            status_color = self.COLORS['RED']
        elif score < 90:
            status_color = self.COLORS['ORANGE']

        # 2. Draw Skeleton (if raw coordinates exist)
        # We use RAW coordinates to match the video feed
        if 'raw_coords' in packet:
            self._draw_skeleton(frame, packet['raw_coords'], faults)

        # 3. Draw Angle Arc (Visual Protractor)
        # We draw this at the knee using the Normalized Angle value
        if 'raw_coords' in packet:
            knee = packet['raw_coords'][25] # Left Knee
            self._draw_angle_text(frame, knee, angle, status_color)

        # 4. Draw HUD (Status Bar)
        self._draw_hud(frame, state, reps, score, status_color)

        # 5. Draw Feedback Toast (Center Screen)
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_hud(self, frame, state, reps, score, color):
        h, w = frame.shape[:2]
        
        # Semi-transparent Top Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), self.COLORS['BLACK'], -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Rep Counter (Left)
        cv2.putText(frame, f"REPS: {reps}", (20, 50), 
                    self.FONT, 1.2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # Score (Center)
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.2, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 50), 
                    self.FONT, 1.2, color, 2, cv2.LINE_AA)

        # State Indicator (Right)
        text_size = cv2.getTextSize(state, self.FONT, 1.0, 2)[0]
        cv2.putText(frame, state, (w - text_size[0] - 20, 50), 
                    self.FONT, 1.0, self.COLORS['BLUE'], 2, cv2.LINE_AA)

    def _draw_toast(self, frame, text, color):
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        
        center_x = (w - text_size[0]) // 2
        center_y = h // 2 # Center of screen

        # Background Box for readability
        pad = 20
        cv2.rectangle(frame, 
                      (center_x - pad, center_y - text_size[1] - pad), 
                      (center_x + text_size[0] + pad, center_y + pad), 
                      self.COLORS['BLACK'], -1)
                      
        cv2.putText(frame, text, (center_x, center_y), 
                    self.FONT, 1.2, color, 3, cv2.LINE_AA)

    def _draw_angle_text(self, frame, knee_point, angle, color):
        """Draws the angle value next to the knee."""
        h, w = frame.shape[:2]
        # Convert normalized coords to pixel coords
        px = int(knee_point.x * w)
        py = int(knee_point.y * h)
        
        text = f"{int(angle)}"
        cv2.putText(frame, text, (px + 20, py), 
                    self.FONT, 0.8, color, 2, cv2.LINE_AA)

    def _draw_skeleton(self, frame, landmarks, faults):
        """Draws stick figure. Highlights joints red if they are faulty."""
        h, w = frame.shape[:2]
        
        # Define Connections: (Start_Index, End_Index)
        connections = [
            (11, 23), (12, 24), # Torso
            (23, 25), (24, 26), # Thighs
            (25, 27), (26, 28), # Shins
            (23, 24)            # Hips
        ]
        
        # 1. Draw Lines (Bones)
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # 2. Draw Joints (Circles)
        # We check if specific faults map to specific joints
        # 11/12 = Shoulders, 23/24 = Hips, 25/26 = Knees, 27/28 = Ankles
        
        joint_indices = [11, 12, 23, 24, 25, 26, 27, 28]
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Determine Color based on Faults
                color = self.COLORS['GREEN']
                
                # Check all faults relevant to this joint
                # If ANY fault is present, turn RED (Priority)
                
                is_faulty = False
                
                # KNEE FAULTS
                if (idx == 25 or idx == 26):
                    if "KNEE_VALGUS" in faults or "SHALLOW" in faults:
                        is_faulty = True
                        
                # ANKLE FAULTS
                if (idx == 27 or idx == 28):
                    if "HEEL_LIFT" in faults:
                        is_faulty = True
                        
                # HIP FAULTS
                if (idx == 23 or idx == 24):
                    if any(f in faults for f in ["UNSTABLE", "ROUNDING", "GOOD_MORNING", "SHALLOW"]):
                        is_faulty = True
                        
                # SHOULDER FAULTS
                if (idx == 11 or idx == 12):
                     if any(f in faults for f in ["ROUNDING", "GOOD_MORNING"]):
                        is_faulty = True

                if is_faulty:
                    color = self.COLORS['RED']

                cv2.circle(frame, (cx, cy), 6, color, -1)
                cv2.circle(frame, (cx, cy), 8, self.COLORS['WHITE'], 1)