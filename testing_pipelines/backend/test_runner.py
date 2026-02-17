# D:\FormCheck-AI\testing_pipelines\backend\test_runner.py
import sys
import os

# 1. Dynamically add the FormCheck-AI root to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

# 2. Try the import!
try:
    from pipelines.pipelines import PipelineFactory
    print("✅ SUCCESS: Testing Studio successfully linked to the core AI Pipelines!")
    
    # Let's test if it actually loads an exercise
    squat_pipeline = PipelineFactory.get_pipeline("squat")
    if squat_pipeline:
        print("✅ SUCCESS: Squat pipeline instantiated.")
except ImportError as e:
    print(f"❌ ERROR: Import binding failed. Details: {e}")