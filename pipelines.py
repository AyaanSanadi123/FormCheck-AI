import cv2
import mediapipe as mp

# Squat Imports
from squat.gatekeeper.gatekeeper import Gatekeeper as SquatGatekeeper
from squat.normalizer.normalizer import SquatNormalizer
from squat.rep.squat_rep import SquatRep
from squat.visualizer.visualizer import Visualizer as SquatVisualizer

# Bench Imports
from flat_barbell_press.gatekeeper.gatekeeper import BenchGatekeeper
from flat_barbell_press.normalizer.normalizer import BenchNormalizer
from flat_barbell_press.rep.rep import BenchPressRep
from flat_barbell_press.visualizer.visualizer import BenchVisualizer

class ExercisePipeline:
    def process(self, frame, landmarks):
        """
        Processes a single frame.
        Args:
            frame: The video frame (image).
            landmarks: Raw MediaPipe landmarks (normalized 0-1).
        Returns:
            The processed frame with overlays.
        """
        raise NotImplementedError

class SquatPipeline(ExercisePipeline):
    def __init__(self):
        self.gatekeeper = SquatGatekeeper()
        self.normalizer = SquatNormalizer()
        self.rep_logic = None
        self.visualizer = SquatVisualizer()
        self.calibration_data = None
        self.status_message = "Initializing..."

    def process(self, frame, landmarks):
        if not landmarks:
            # If no landmarks, just draw the frame (maybe add "No user" text later)
            return frame

        # 1. Gatekeeper (Calibration Phase)
        if not self.rep_logic:
            passed, msg, cal_data = self.gatekeeper.check(landmarks)
            self.status_message = msg
            
            # Draw Gatekeeper Overlay (Simple text for now)
            cv2.putText(frame, f"CALIBRATION: {msg}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if passed:
                self.calibration_data = cal_data
                self.rep_logic = SquatRep(cal_data)
                print(f"Squat Calibration Complete: {cal_data}")
            
            return frame

        # 2. Normalization
        # SquatRep expects normalized landmarks for geometry, but raw for floor checks
        normalized_landmarks = self.normalizer.process(landmarks)

        # 3. Rep Logic
        # We pass BOTH normalized and raw landmarks
        packet = self.rep_logic.process(normalized_landmarks, raw_landmarks=landmarks)

        # 4. Visualization
        # Visualizer expects the full packet
        return self.visualizer.draw(frame, packet)

class BenchPipeline(ExercisePipeline):
    def __init__(self):
        self.gatekeeper = BenchGatekeeper()
        self.normalizer = BenchNormalizer()
        self.rep_logic = None
        self.visualizer = BenchVisualizer()
        self.calibration_data = None
        self.status_message = "Initializing..."

    def process(self, frame, landmarks):
        if not landmarks:
            return frame

        # 1. Gatekeeper
        if not self.rep_logic:
            passed, msg, cal_data = self.gatekeeper.check(landmarks)
            
            cv2.putText(frame, f"SETUP: {msg}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if passed:
                self.calibration_data = cal_data
                self.rep_logic = BenchPressRep(cal_data)
                print(f"Bench Calibration Complete: {cal_data}")
            
            return frame

        # 2. Normalization
        # BenchRep accepts both normalized and raw landmarks; raw landmarks are passed for visualization.
        normalized_landmarks = self.normalizer.process(landmarks)

        # 3. Rep Logic
        packet = self.rep_logic.process(normalized_landmarks, raw_landmarks=landmarks)

        # 4. Visualization
        return self.visualizer.draw(frame, packet)

class PipelineFactory:
    @staticmethod
    def get_pipeline(exercise_name):
        if exercise_name.lower() == 'squat':
            return SquatPipeline()
        elif exercise_name.lower() in ['bench', 'bench_press', 'flat_barbell_press']:
            return BenchPipeline()
        else:
            raise ValueError(f"Unknown exercise: {exercise_name}")
