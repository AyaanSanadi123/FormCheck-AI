# FORMCHECK-AI: TESTING ARCHITECTURE BLUEPRINT
# Version: 1.0
# Goal: Build a "Time-Travel" Web Studio for frame-by-frame computer vision debugging.

## 1. THE TECH STACK
* **Frontend Framework:** Next.js (React)
* **UI Component Library:** shadcn/ui (For sleek, accessible, and fast dashboard components)
* **Data Visualization:** Recharts (or similar React charting library for kinematics)
* **Backend Framework:** FastAPI (Python)
* **Computer Vision Core:** OpenCV & MediaPipe (Imported from the main pipeline)
* **Cloud Integration:** Google Drive API / `gdown` (For fetching test videos)

---

## 2. THE DIRECTORY STRUCTURE (D: Drive)
To maintain strict separation of concerns while allowing clean Python imports, the project will be restructured as follows:

D:\FormCheck-AI\
├── pipelines\                 # The Production AI Core (Untouched by testing)
│   ├── squat\
│   ├── dips\
│   └── pipelines.py           # The PipelineFactory
│
└── testing_pipelines\         # The Debugging Studio
    ├── backend\               # Python / FastAPI server
    │   ├── local_cache\       # Temporary storage for downloaded Drive videos
    │   ├── telemetry_data\    # Where the massive JSON output files live
    │   └── test_runner.py     # The extraction engine
    │
    └── frontend\              # Next.js Application
        ├── components\        # shadcn/ui components (tabs, sliders, cards)
        ├── public\            # Static assets
        └── app\               # Next.js pages and layouts

---

## 3. PHASED IMPLEMENTATION PLAN

### Phase 1: Architecture Restructure & Import Binding
* **Objective:** Establish the workspace and prove the frontend, backend, and core AI can communicate.
* **Tasks:**
  1. Reorganize the `FormCheck-AI` folder in the D drive into the `pipelines` and `testing_pipelines` structure.
  2. Write a minimal Python test script in `testing_pipelines/backend` to successfully import `PipelineFactory` from the sibling `pipelines` folder.
  3. Initialize the Next.js app with `shadcn/ui` installed in the `frontend` folder.
  4. Initialize a basic FastAPI server in the `backend` folder.

### Phase 2: The Telemetry Extraction Engine (Backend)
* **Objective:** Build the Python pipeline that converts raw Google Drive videos into structured JSON data.
* **Tasks:**
  1. **Drive Fetcher:** Implement a function using a Google Drive URL to securely download `.mp4` files into the `local_cache` directory.
  2. **The Extraction Loop:** Run the downloaded video through the AI pipeline frame-by-frame (without rendering a pop-up window).
  3. **State Logging:** On every frame, extract the Gatekeeper's calibration matrix, the Normalizer's canonical coordinates, and the Rep Logic's metrics (angles, velocities, fault states).
  4. **The Serializer:** Write the custom JSON encoder to safely dump the NumPy arrays and dictionaries into a `telemetry.json` file.

### Phase 3: The Dashboard Scaffold & Video Sync (Frontend)
* **Objective:** Build the core Next.js interface to bind the video playback to the JSON data.
* **Tasks:**
  1. **Layout Design:** Use `shadcn/ui` to build a clean split-screen dashboard (Video Player on the left, Data Panels on the right).
  2. **Data Ingestion:** Create API routes to fetch the `telemetry.json` payload from the FastAPI backend into the React state.
  3. **The Sync Engine:** Tie an HTML5 video player's `currentTime` to a master timeline scrubber. As the video plays (or is manually scrubbed), React dynamically filters the JSON array to display the exact data object for that specific millisecond.

### Phase 4: The X-Ray Visualizers (Frontend)
* **Objective:** Translate the raw JSON into intuitive debugging graphics.
* **Tasks:**
  1. **Kinematics Chart (Recharts):** Create a live line graph plotting joint angles and velocity over time. A vertical playhead line moves across the chart in sync with the video.
  2. **The Canonical Viewer (2D Scatter Plot):** Create a visual grid that renders the `normalized_coords`. This allows us to watch the user move around the strictly enforced `(0,0)` origin, proving the perspective flip and scaling worked.
  3. **State & Fault Logger:** Use `shadcn/ui` alert cards to flash when specific pipeline states change (e.g., transitioning from `IDLE` to `ECCENTRIC` or triggering `PENDULUM_SWING`).

### Phase 5: Test Library Execution
* **Objective:** Run the engine against real-world edge cases.
* **Tasks:**
  1. Curate a Google Drive folder containing videos of perfect reps, shallow depth, bad camera angles, and intentional cheating for all exercises.
  2. Run the batch through the backend to generate the telemetry logs.
  3. Use the Next.js frontend to visually hunt down and patch any false positives or false negatives in the core math.