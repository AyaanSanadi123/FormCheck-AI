import numpy as np

class HammerCurlNormalizer:
    def __init__(self):
        self.alpha = 0.25
        self.smoothed_angle = 0

    def process(self, landmarks):
        if not landmarks: return []
        
        # Pivot at Hip (24), Align Shoulder (12)
        hip, sh = landmarks[24], landmarks[12]
        
        # Calculate lean relative to vertical
        curr_lean = np.degrees(np.arctan2(sh.x - hip.x, sh.y - hip.y))
        # We want to rotate until the lean is 180 (straight up from hip)
        delta = 180 - curr_lean
        
        self.smoothed_angle = (self.alpha * delta) + (1 - self.alpha) * self.smoothed_angle
        return self._apply_rotation(landmarks, self.smoothed_angle, hip.x, hip.y)

    def _apply_rotation(self, landmarks, angle_deg, px, py):
        rad = np.radians(angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        
        normalized = []
        for lm in landmarks:
            nx = c * (lm.x - px) - s * (lm.y - py) + px
            ny = s * (lm.x - px) + c * (lm.y - py) + py
            normalized.append(type(lm)(x=nx, y=ny, z=lm.z, visibility=lm.visibility))
        return normalized