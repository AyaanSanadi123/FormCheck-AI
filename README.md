# FormCheck AI

An AI-powered personal trainer that analyzes your exercise form in real-time using computer vision.

## Supported Exercises
- **Barbell Rows**
- **Deadlift**
- **Dips**
- **Dumbbell Rows**
- **Flat Barbell Press**
- **Hamstring Curls**
- **Incline Press**
- **Lat Pulldowns**
- **Seated Row**
- **Skull Crusher**
- **Squat**
- **Tricep Kickbacks**
- **Tricep Pushdowns**

## Requirements
- Python 3.10+
- Webcam

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AyaanSanadi123/FormCheck-AI.git
   cd FormCheck-AI
   ```

2. **Create and Activate a Virtual Environment:**
   It is highly recommended to use a virtual environment to manage dependencies.

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application and specify the exercise you want to perform. All exercise logic is now located in the `pipelines/` directory.

**General Command:**
```bash
python pipelines/main.py --exercise [EXERCISE_NAME]
```

**Example (Squat):**
```bash
python pipelines/main.py --exercise squat
```

**Using a Video File:**
You can analyze a pre-recorded video by providing the path to the file.
```bash
python pipelines/main.py --exercise squat --source path/to/video.mp4
```

## How It Works
1. **Gatekeeper:** Ensures you are in the correct starting position and camera angle before starting.
2. **Normalizer:** "Virtual Camera" that rotates your skeleton to a perfect side view for accurate angle measurement.
3. **Rep Logic:** State machine that counts reps and detects specific form faults.
4. **Visualizer:** Draws real-time feedback, skeleton overlays, and rep counters on the screen.

## Project Structure
- `pipelines/`: Contains the core logic for all supported exercises.
- `testing_pipelines/`: Contains the automated testing architecture for verifying form detection.

## Controls
- Press **'q'** to quit the application.
