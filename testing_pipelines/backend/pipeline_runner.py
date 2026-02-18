import cv2
import mediapipe as mp
import json
import numpy as np
import os
import math
from typing import Tuple

# Dynamically import your PipelineFactory from the sibling directory
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

try:
    from pipelines.pipelines import PipelineFactory
except ImportError:
    print("Warning: Could not import PipelineFactory. Ensure your folder structure is correct.")


class NpEncoder(json.JSONEncoder):
    """
    Sanitizes NumPy arrays, floats, and custom Landmark objects for JSON.
    Crucially converts NaN and Inf to null (None) to prevent Next.js crashes.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Edge Case 4: The NaN Math Crash
            if np.isnan(obj) or np.isinf(obj):
                return None # Becomes 'null' in JSON, which JS parsers handle perfectly
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle our custom Landmark objects from the Normalizer
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            return {
                "x": round(float(obj.x), 4), 
                "y": round(float(obj.y), 4), 
                "z": round(float(getattr(obj, 'z', 0.0)), 4), 
                "visibility": round(float(getattr(obj, 'visibility', 1.0)), 4)
            }
            
        return super(NpEncoder, self).default(obj)


def process_video_headless(video_path: str, exercise_name: str, output_dir: str) -> Tuple[bool, str]:
    """
    Runs the AI pipeline on a video without rendering a UI.
    Extracts frame-by-frame telemetry and saves it as a JSON file.
    """
    print(f"Starting headless extraction for: {video_path}")
    
    # Extract filename without extension (e.g., "local_cache/1.mp4" -> "1")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json_path = os.path.join(output_dir, f"{base_name}_telemetry.json")

    # Initialize OpenCV and MediaPipe
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, f"Failed to open video file: {video_path}"

    # Edge Case 3: Variable Video Framerates (The Velocity Trap)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps): 
        fps = 30.0 # Fallback for corrupted metadata

    # --- REVERTED: Use standard MediaPipe initialization ---
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Initialize your core AI Pipeline
    pipeline = PipelineFactory.get_pipeline(exercise_name)
    if not pipeline:
        return False, f"Unknown exercise pipeline: {exercise_name}"

    telemetry_log = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / fps
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image_rgb)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        # Base Frame Dictionary
        frame_data = {
            "frame_id": frame_count,
            "timestamp": round(timestamp, 3),
            "gatekeeper": {"calibrated": False},
            "rep_logic": None,
            "normalized_coords": None
        }

        # Edge Case 1: The "Ghost" Frame (Missing Landmarks)
        if not landmarks:
            frame_data["status"] = "NO_DETECTION"
            telemetry_log.append(frame_data)
            continue

        frame_data["status"] = "TRACKING"

        # Pass to the AI Pipeline silently (no imshow)
        pipeline.process(frame, landmarks)
        
        # 1. Extract Gatekeeper State (Edge Case 2: Warmup Lag)
        if getattr(pipeline, 'calibration_data', None):
            frame_data["gatekeeper"] = {
                "calibrated": True,
                "active_side": pipeline.calibration_data.get('active_side'),
                "scale_factor": round(pipeline.calibration_data.get('scale_factor', 1.0), 4)
            }
        else:
            frame_data["gatekeeper"]["status"] = "CALIBRATING"

        # 2. Extract Normalizer & Rep Logic State
        if getattr(pipeline, 'rep_logic', None):
            # Snoop the normalized coordinates directly from the normalizer memory
            norm_landmarks = pipeline.normalizer.process(landmarks, pipeline.calibration_data)
            frame_data["normalized_coords"] = norm_landmarks
            
            # Snoop the active rep logic packet
            packet = pipeline.rep_logic.process(norm_landmarks, raw_landmarks=landmarks, timestamp=timestamp)
            if packet:
                # Strip massive raw arrays out to prevent the JSON file from becoming 50MB
                packet.pop('raw_coords', None) 
                packet.pop('coords', None)
                frame_data["rep_logic"] = packet

        telemetry_log.append(frame_data)

    cap.release()
    mp_pose.close()

    # Serialize and Save using our custom Crash-Proof Encoder
    try:
        with open(output_json_path, 'w') as f:
            json.dump(telemetry_log, f, cls=NpEncoder, indent=2)
        return True, f"Successfully exported {frame_count} frames to {output_json_path}"
    except Exception as e:
        return False, f"JSON Serialization Error: {e}"