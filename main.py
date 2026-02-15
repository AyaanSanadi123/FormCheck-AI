import cv2
import mediapipe as mp
import argparse
import sys
from pipelines import PipelineFactory

def main():
    parser = argparse.ArgumentParser(description="AI Personal Trainer - Form Correction")
    parser.add_argument('--exercise', type=str, required=True, 
                        help="Name of exercise (squat, bench)")
    parser.add_argument('--source', type=str, default="0", 
                        help="Video source: '0' for webcam or path to video file")
    
    args = parser.parse_args()

    # 1. Initialize Pipeline
    try:
        pipeline = PipelineFactory.get_pipeline(args.exercise)
        print(f"Loaded pipeline for: {args.exercise}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Setup Video Source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    # 3. Setup MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    print("Starting... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video finished or stream failed.")
            break

        # MediaPipe requires RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Pass Landmarks to Pipeline
        # results.pose_landmarks.landmark is a list of NormalizedLandmark objects
        landmarks = None
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
        
        # Capture current time for velocity calculations
        import time
        timestamp = time.time()

        # Process Frame
        # The pipeline handles logic + drawing
        output_frame = pipeline.process(frame, landmarks, timestamp=timestamp)

        # Show Result
        cv2.imshow('FormCheck AI', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
