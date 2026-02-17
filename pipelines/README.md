# FormCheck AI

An AI-powered personal trainer that analyzes your exercise form in real-time using computer vision.

## Supported Exercises
- **Squat**
- **Bench Press (Flat Barbell)**

## Requirements
- Python 3.8+
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
   python3 -m venv venv
   source venv/bin/activate
   ```

   **Windows:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application and specify the exercise you want to perform.

**Squat:**
```bash
python main.py --exercise squat
```

**Bench Press:**
```bash
python main.py --exercise bench
```

**Using a Video File:**
You can analyze a pre-recorded video by providing the path to the file.
```bash
python main.py --exercise squat --source path/to/video.mp4
```

## How It Works
1. **Gatekeeper:** Ensures you are in the correct starting position and camera angle before starting.
2. **Normalizer:** "Virtual Camera" that rotates your skeleton to a perfect side view for accurate angle measurement.
3. **Rep Logic:** State machine that counts reps and detects specific form faults (e.g., "Not deep enough", "Heels lifting").
4. **Visualizer:** Draws real-time feedback, skeleton overlays, and rep counters on the screen.

## Controls
- Press **'q'** to quit the application.
