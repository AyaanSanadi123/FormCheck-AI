# D:\FormCheck-AI\testing_pipelines\backend\main.py
import os
import glob
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Import the Queue Manager we architected
from queue_manager import QueueManager

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="FormCheck-AI Testing API")

# --- CORS CONFIGURATION ---
origins = [
    os.getenv("FRONTEND_URL", "http://localhost:3000")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# --- MEDIA SERVER CONFIGURATION ---
# Ensure the directory exists before mounting to prevent startup crashes
os.makedirs("local_cache", exist_ok=True)
os.makedirs("telemetry_data", exist_ok=True)

# Tell FastAPI to serve the local_cache folder as a media directory
app.mount("/media", StaticFiles(directory="local_cache"), name="media")

# Initialize our core components
queue_manager = QueueManager(
    cache_dir="local_cache", 
    telemetry_dir="telemetry_data", 
    batch_size=5
)

# --- REQUEST MODELS ---
class PeekPayload(BaseModel):
    drive_folder_url: str

class TestRunPayload(BaseModel):
    drive_folder_url: str
    exercise_name: str
    batch_index: int = 0  # Default to 0 (Batch 1)

# --- API ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "FormCheck-AI Backend is Online"}

@app.post("/api/test/peek")
def peek_folder(payload: PeekPayload):
    """
    The Scout. Checks how many videos are in the folder without downloading them.
    """
    try:
        from drive_fetcher import get_video_list
        # Fetch the array of IDs
        all_file_ids = get_video_list(payload.drive_folder_url)
        
        return {
            "status": "success", 
            "data": {
                "total_videos": len(all_file_ids),
                "batch_size": queue_manager.batch_size
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test/run")
def trigger_test_batch(payload: TestRunPayload):
    """
    Receives the Drive URL and batch index, and orchestrates the AI processing.
    """
    try:
        # Hand off to our Queue Manager with the specific batch index
        results = queue_manager.process_folder(
            drive_folder_url=payload.drive_folder_url, 
            exercise_name=payload.exercise_name,
            batch_index=payload.batch_index
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/results/{video_name}")
def get_telemetry_data(video_name: str):
    """
    Next.js calls this to fetch the actual JSON data for charting.
    Example: GET /api/test/results/1 
    """
    file_name = f"{video_name}_telemetry.json" 
    file_path = os.path.join(queue_manager.telemetry_dir, file_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Telemetry data not found for this video.")

    # FileResponse optimally streams the JSON file directly to the browser
    return FileResponse(file_path, media_type="application/json")

@app.delete("/api/test/cache")
def sweep_cache():
    """
    The Self-Destruct Sequence. Deletes all MP4s and JSONs to free up disk space.
    """
    try:
        # 1. Sweep the MP4 local_cache
        for f in glob.glob(os.path.join(queue_manager.cache_dir, "*")):
            if os.path.isfile(f):
                os.remove(f)
            
        # 2. Sweep the JSON telemetry_data
        for f in glob.glob(os.path.join(queue_manager.telemetry_dir, "*")):
            if os.path.isfile(f):
                os.remove(f)
            
        return {"status": "success", "message": "Cache swept clean."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# --- SERVER STARTUP ---
if __name__ == "__main__":
    # Dynamically pull the port from the .env file
    port = int(os.getenv("API_PORT", 8000))
    print(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)