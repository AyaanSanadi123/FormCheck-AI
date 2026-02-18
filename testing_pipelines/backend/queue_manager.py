# D:\FormCheck-AI\testing_pipelines\backend\queue_manager.py
import os
from typing import List, Dict
import traceback

# Import the actual modules we built
from drive_fetcher import get_video_list, download_batch
from pipeline_runner import process_video_headless

class QueueManager:
    """
    Orchestrates the downloading, batch processing, and cleanup of test videos.
    Now supports paginated batching (Batch 1, Batch 2, etc.) for massive folders.
    """
    def __init__(self, cache_dir: str = "local_cache", telemetry_dir: str = "telemetry_data", batch_size: int = 5):
        self.cache_dir = cache_dir
        self.telemetry_dir = telemetry_dir
        self.batch_size = batch_size
        
        # Ensure our directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.telemetry_dir, exist_ok=True)

    def process_folder(self, drive_folder_url: str, exercise_name: str, batch_index: int = 0) -> Dict:
        """The main entry point for processing a specific batch of videos."""
        
        print(f"Scanning Google Drive Folder: {drive_folder_url}")
        
        # 1. Fetch ALL valid video IDs from the cloud
        all_file_ids = get_video_list(drive_folder_url)
        total_videos = len(all_file_ids)
        
        if total_videos == 0:
            print("No valid videos found in folder.")
            return {"successful": 0, "failed": 0, "total_videos": 0}

        # --- THE PAGINATION LOGIC ---
        start_idx = batch_index * self.batch_size
        end_idx = start_idx + self.batch_size
        
        # Slice the array to get only the IDs for this specific batch
        target_file_ids = all_file_ids[start_idx:end_idx]
        
        if not target_file_ids:
            print(f"Batch {batch_index + 1} is empty (out of bounds).")
            return {"successful": 0, "failed": 0, "total_videos": total_videos}

        print(f"Processing Batch {batch_index + 1} (Videos {start_idx + 1} to {min(end_idx, total_videos)} of {total_videos})...")

        results = {"successful": 0, "failed": 0, "details": [], "total_videos": total_videos}
        
        try:
            # 2. Download the specific batch
            downloaded_files = download_batch(target_file_ids, self.cache_dir)
            
            # 3. Process the batch headless
            for video_path in downloaded_files:
                print(f"Running headless AI on: {video_path}")
                success, msg = process_video_headless(video_path, exercise_name, self.telemetry_dir)
                
                # Log the result
                if success:
                    results["successful"] += 1
                    results["details"].append({"file": video_path, "status": "Processed", "message": msg})
                else:
                    results["failed"] += 1
                    results["details"].append({"file": video_path, "status": "Failed", "message": msg})
                
        except Exception as e:
            # Capture the massive, detailed stack trace
            full_trace = traceback.format_exc() 
            
            print(f"CRITICAL ERROR in Batch {batch_index + 1}:")
            print(full_trace) # Print it to the Python terminal
            
            # Send the full stack trace to the Next.js UI!
            results["critical_error"] = full_trace
            
        finally:
            # THE CRASH NET (Cleanup)
            # We purposely leave this commented out in the Testing Studio environment
            # so the Next.js frontend can stream the files. The "Sweep Cache" button handles deletion.
            # print(f"Emptying local cache for Batch {batch_index + 1}...")
            # self._empty_cache()
            pass

        print("\n✅ Queue Processing Complete!")
        return results

    def _empty_cache(self):
        """Forcibly deletes all files in the local cache directory."""
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# If you want to test this script directly:
if __name__ == "__main__":
    manager = QueueManager()
    summary = manager.process_folder("YOUR_DRIVE_URL_HERE", "dips", batch_index=0)
    print(summary)