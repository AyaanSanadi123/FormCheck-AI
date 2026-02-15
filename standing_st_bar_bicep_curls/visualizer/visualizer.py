import cv2
import numpy as np
from collections import deque
import mediapipe as mp # Import for MediaPipe PoseLandmark indices

class BicepCurlVisualizer:
    def __init__(self):
        # Color Palette (B, G, R) - OpenCV uses BGR
        self.COLORS = {
            'GREEN': (0, 255, 0),      # Good/Pass
            'ORANGE': (0, 165, 255),   # Warning (Blueprint standard)
            'RED': (0, 0, 255),        # Bad/Fail
            'WHITE': (255, 255, 255),  # Text/HUD
            'BLACK': (0, 0, 0),        # Background
            'CYAN': (255, 255, 0)      # General Info
        }
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        self.MP_POSE = mp.solutions.pose.PoseLandmark

        # Visual History for Bar Path (Stores last 20 frames of Wrist coords)
        self.path_history = deque(maxlen=20) 

    def draw(self, frame, packet):
        """
        Main rendering pipeline, compliant with blueprint.
        Args:
            frame: Raw video frame.
            packet: Output from RepLogic.process().
        """
        if packet is None:
            # If no packet, return frame or display a "Searching..." message
            h, w = frame.shape[:2]
            cv2.putText(frame, "SEARCHING...", (w // 2 - 100, h // 2), 
                        self.FONT, 1, self.COLORS['WHITE'], 2)
            return frame

        # Unpack Data
        state = packet.get('state', "IDLE")
        score = packet.get('score', 100)
        reps = packet.get('reps', 0)
        feedback = packet.get('feedback', "")
        faults = packet.get('faults', [])
        raw_coords = packet.get('raw_coords', None)
        metrics = packet.get('metrics', {}) # Get the metrics dictionary
        
        # Extract angle from metrics
        angle = metrics.get('angle', 0)

        # 1. Determine Status Color (Blueprint compliant)
        status_color = self.COLORS['GREEN']
        if score < 70: status_color = self.COLORS['RED']
        elif score < 90: status_color = self.COLORS['ORANGE']

        if raw_coords:
            # 2. Draw Skeleton & Joints
            self._draw_skeleton(frame, raw_coords, faults, packet.get('active_side', 'RIGHT'))
            
            # 3. Draw Flexion Gauge / Angle Text (using raw_coords for position)
            self._draw_flexion_gauge(frame, raw_coords, angle, packet.get('active_side', 'RIGHT'))

            # 4. Draw Elbow Anchor (using raw_coords for position)
            self._draw_elbow_anchor(frame, raw_coords, faults, packet.get('active_side', 'RIGHT'))

            # 5. Draw Spine Integrity Line
            self._draw_spine_integrity(frame, raw_coords, faults)

        # 6. Draw HUD (Top Bar)
        self._draw_hud(frame, state, reps, score, status_color)

        # 7. Draw Feedback Toast
        if feedback:
            self._draw_toast(frame, feedback, status_color)

        return frame

    def _draw_skeleton(self, frame, landmarks, faults, active_side):
        """Draws stick figure for Bicep Curl, highlighting active arm."""
        h, w = frame.shape[:2]
        
        # Determine which side is active for primary drawing
        active_shoulder_idx = self.MP_POSE.RIGHT_SHOULDER.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_SHOULDER.value
        active_elbow_idx = self.MP_POSE.RIGHT_ELBOW.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_ELBOW.value
        active_wrist_idx = self.MP_POSE.RIGHT_WRIST.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_WRIST.value
        active_hip_idx = self.MP_POSE.RIGHT_HIP.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_HIP.value

        # Connections for a side view, focusing on the arm and torso
        connections = [
            (active_shoulder_idx, active_elbow_idx),
            (active_elbow_idx, active_wrist_idx),
            (active_hip_idx, active_shoulder_idx), # Torso connection
            (self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value),
            (self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value)
        ]
        
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start.visibility > 0.5 and end.visibility > 0.5:
                p1 = (int(start.x * w), int(start.y * h))
                p2 = (int(end.x * w), int(end.y * h))
                # Default to white, can be changed later for specific highlights
                cv2.line(frame, p1, p2, self.COLORS['WHITE'], 2, cv2.LINE_AA)

        # Draw Joints and apply fault-based coloring
        joint_indices = [
            self.MP_POSE.LEFT_SHOULDER.value, self.MP_POSE.RIGHT_SHOULDER.value,
            self.MP_POSE.LEFT_ELBOW.value, self.MP_POSE.RIGHT_ELBOW.value,
            self.MP_POSE.LEFT_WRIST.value, self.MP_POSE.RIGHT_WRIST.value,
            self.MP_POSE.LEFT_HIP.value, self.MP_POSE.RIGHT_HIP.value
        ]
        
        for idx in joint_indices:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                color = self.COLORS['GREEN'] # Default
                
                # Fault-specific coloring (example: ELBOW_SWAY affecting elbow)
                if "ELBOW_SWAY" in faults and idx == active_elbow_idx:
                    color = self.COLORS['RED']
                elif "BACK_LEAN" in faults and (idx == active_shoulder_idx or idx == active_hip_idx):
                    color = self.COLORS['RED']

                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 8, self.COLORS['WHITE'], 1) # Outline

    def _draw_elbow_anchor(self, frame, landmarks, faults, active_side):
        """Draws a fixed target zone for the elbow."""
        h, w = frame.shape[:2]
        active_elbow_idx = self.MP_POSE.RIGHT_ELBOW.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_ELBOW.value
        elb = landmarks[active_elbow_idx]
        center = (int(elb.x * w), int(elb.y * h))
        
        color = self.COLORS['CYAN'] # Changed from PIVOT to CYAN for consistency
        if "ELBOW_SWAY" in faults: 
            color = self.COLORS['RED']
            cv2.circle(frame, center, 15, color, 2) # Highlight bigger for fault
        
        cv2.drawMarker(frame, center, color, cv2.MARKER_CROSS, 20, 2)

    def _draw_flexion_gauge(self, frame, landmarks, angle, active_side):
        """Draws the angle text near the elbow."""
        h, w = frame.shape[:2]
        active_elbow_idx = self.MP_POSE.RIGHT_ELBOW.value if active_side == 'RIGHT' else self.MP_POSE.LEFT_ELBOW.value
        elb_x, elb_y = int(landmarks[active_elbow_idx].x * w), int(landmarks[active_elbow_idx].y * h)
        
        color = self.COLORS['GREEN'] if angle < 60 else self.COLORS['WHITE'] # Example condition
        
        cv2.putText(frame, f"{int(angle)}deg", (elb_x + 20, elb_y - 20), 
                    self.FONT, 0.7, color, 2)

    def _draw_spine_integrity(self, frame, landmarks, faults):
        """Draws a vertical reference line to highlight back lean."""
        h, w = frame.shape[:2]
        sh = landmarks[self.MP_POSE.RIGHT_SHOULDER.value] # Assuming right side is active
        hip = landmarks[self.MP_POSE.RIGHT_HIP.value] # Assuming right side is active
        
        p1 = (int(sh.x * w), int(sh.y * h))
        p2 = (int(hip.x * w), int(hip.y * h))

        color = self.COLORS['WHITE']
        if "BACK_LEAN" in faults:
            color = self.COLORS['RED']
            # Draw an arrow indicating the direction of cheat
            cv2.arrowedLine(frame, p1, (p1[0] + 40, p1[1]), color, 3) # Example arrow
        
        if sh.visibility > 0.5 and hip.visibility > 0.5:
            cv2.line(frame, p1, p2, color, 3)

    def _draw_hud(self, frame, state, reps, score, color):
        """Draws the top status bar, blueprint compliant."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), self.COLORS['BLACK'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Reps (Left)
        cv2.putText(frame, f"REPS: {reps}", (20, 40), 
                    self.FONT, 1.0, self.COLORS['WHITE'], 2)
        
        # Score (Center)
        score_text = f"SCORE: {score}"
        text_size = cv2.getTextSize(score_text, self.FONT, 1.0, 2)[0]
        center_x = (w - text_size[0]) // 2
        cv2.putText(frame, score_text, (center_x, 40), 
                    self.FONT, 1.0, color, 2)

        # State (Right)
        state_text_size = cv2.getTextSize(state, self.FONT, 0.8, 2)[0]
        cv2.putText(frame, state, (w - state_text_size[0] - 20, 40), # 20px padding from right
                    self.FONT, 0.8, self.COLORS['CYAN'], 2)

    def _draw_toast(self, frame, text, color):
        """Draws a floating message box in the center, blueprint compliant."""
        h, w = frame.shape[:2]
        text_size = cv2.getTextSize(text, self.FONT, 1.2, 3)[0]
        center_x = (w - text_size[0]) // 2
        center_y = h // 2
        
        pad = 20
        cv2.rectangle(frame, 
                      (center_x - pad, center_y - text_size[1] - pad), 
                      (center_x + text_size[0] + pad, center_y + pad), 
                      self.COLORS['BLACK'], -1)
        
        cv2.putText(frame, text, (center_x, center_y), 
                    self.FONT, 1.2, color, 3)