# D:\FormCheck-AI\testing_pipelines\backend\drive_fetcher.py
import re
import os
import requests
import gdown
from typing import List
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

def extract_folder_id(url: str) -> str:
    """Uses Regex to cleanly extract the Drive ID from any messy Google URL."""
    match = re.search(r'[-\w]{25,}', url)
    if not match:
        raise ValueError("Invalid Google Drive URL. Could not extract Folder ID.")
    return match.group(0)

def get_video_list(folder_url: str) -> List[str]:
    """
    The 'Smart Peek'. Queries the Drive API to get file IDs inside the folder.
    Strictly filters out non-video formats to prevent bad downloads.
    """
    # Securely retrieve the API key from the environment
    api_key = os.getenv("GOOGLE_DRIVE_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL: GOOGLE_DRIVE_API_KEY is missing from the .env file.")

    folder_id = extract_folder_id(folder_url)
    print(f"Checking Google Drive Folder ID: {folder_id}...")

    api_url = "https://www.googleapis.com/drive/v3/files"
    query = f"'{folder_id}' in parents and mimeType contains 'video/' and trashed=false"
    
    params = {
        'q': query,
        'key': api_key,
        'fields': 'files(id, name, mimeType)',
        'pageSize': 100
    }

    response = requests.get(api_url, params=params)
    
    if response.status_code != 200:
        raise ConnectionError(f"Drive API Error: {response.json()}")

    files = response.json().get('files', [])
    print(f"Validated {len(files)} video(s) in the folder.")
    
    return [file['id'] for file in files]

def download_batch(file_ids: List[str], dest_dir: str) -> List[str]:
    """
    Downloads a batch of files directly into the local cache.
    Automatically renames them to sequential integers (1.mp4, 2.mp4, etc.)
    """
    downloaded_paths = []
    
    for index, file_id in enumerate(file_ids):
        safe_name = f"{index + 1}.mp4"
        dest_path = os.path.join(dest_dir, safe_name)
        
        print(f"Downloading Video {index + 1}/{len(file_ids)}...")
        
        try:
            gdown.download(id=file_id, output=dest_path, quiet=True)
            downloaded_paths.append(dest_path)
        except Exception as e:
            print(f"Failed to download File ID {file_id}: {e}")
            
    return downloaded_paths