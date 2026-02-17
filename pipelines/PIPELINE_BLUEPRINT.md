# AI Personal Trainer - Pipeline Blueprint & Standards

This document serves as the architectural standard for implementing new exercises in the FormCheck-AI project. All new exercise modules must adhere to these structural requirements to ensure compatibility with the main application orchestration.

## 1. Directory Structure

Each exercise is a self-contained Python package located in the root directory.

```
exercise-name/
├── gatekeeper/
│   ├── gatekeeper.py       # Class: Gatekeeper
│   └── gatekeeper.txt      # (Optional) Prompt/Logic documentation
├── normalizer/
│   ├── normalizer.py       # Class: Normalizer (or ExerciseNormalizer)
│   └── normalizer.txt
├── rep/
│   ├── rep.py              # Class: RepLogic (or ExerciseRep)
│   └── rep.txt
└── visualizer/
    ├── visualizer.py       # Class: Visualizer
    └── visualizer.txt
```

**Naming Convention:**
- Folder names: Lowercase, hyphen-separated (e.g., `seated-row`, `deadlift`).
- File names: Prefer generic names (`gatekeeper.py`, `rep.py`) inside the specific folders.
- Class names: PascalCase. Can be generic (`Gatekeeper`) or specific (`SquatGatekeeper`). The `PipelineFactory` will look for standard names first.

---

## 2. Component Interfaces

The pipeline consists of four sequential stages. Data flows as follows:
`Video Frame` -> `Gatekeeper` -> `Normalizer` -> `Rep Logic` -> `Visualizer` -> `Output Frame`

### A. Gatekeeper (`gatekeeper.py`)

**Purpose:** 
Verifies the user is in the correct starting position, visible, and stable before exercise tracking begins. It calculates initial body proportions (calibration) used by subsequent stages.

**Requirements:**
1.  **Method:** `check(self, landmarks)`
    *   **Input:** Raw MediaPipe landmarks (List).
    *   **Output:** Tuple `(passed: bool, message: str, calibration_data: dict)`.
2.  **Logic:**
    *   **Detection:** Check if a user is present.
    *   **Active Side:** Determine which side of the body is facing the camera (Left/Right) based on hip visibility.
    *   **Visibility:** Ensure ALL critical joints for the exercise are visible (visibility > threshold).
    *   **Geometry:** Check basic alignment (e.g., "Stand up straight", "Step back").
    *   **Stability:** specific buffer (e.g., 30-60 frames) to ensure the user is holding still before passing.
3.  **Calibration Data (Dict):**
    *   Must include: `active_side` ("LEFT" or "RIGHT").
    *   Should include: `scale_factor` (e.g., Torso Length), `floor_y` (Y-coordinate of lowest point).

### B. Normalizer (`normalizer.py`)

**Purpose:**
Standardizes the raw landmarks into a "Canonical Pose" to simplify math in the Rep Logic. This ensures the logic works regardless of camera distance, height, or facing direction.

**Requirements:**
1.  **Method:** `process(self, landmarks, calibration_data)`
    *   **Input:** Raw landmarks, Calibration dict from Gatekeeper.
    *   **Output:** List of normalized `Landmark` objects (with `.x`, `.y`, `.z`, `.visibility`).
2.  **Standardization Rules:**
    *   **Origin:** Translate the **Primary Mechanical Pivot** to `(0, 0)`. (e.g., Hip for Squat, Knee for Leg Extension, Shoulder for Pushdown).
    *   **Facing:** Flip X-coordinates if necessary so the user always faces **RIGHT** (positive X is forward).
    *   **Scale:** Divide coordinates by `calibration_data['scale_factor']` (e.g., Torso Length = 1.0).
    *   **Floor:** If applicable, align floor to `Y=0`.

### C. Rep Logic (`rep.py`)

**Purpose:**
The "Brain" of the exercise. It tracks the repetition state machine, counts reps, calculates scores, and identifies form faults.

**Requirements:**
1.  **Init:** `__init__(self, calibration_data)`
    *   Receive baselines established by the Gatekeeper.
2.  **Method:** `process(self, landmarks, raw_landmarks=None, timestamp=None)`
    *   **Input:** Normalized landmarks (for logic), Raw landmarks (passed through for drawing), Timestamp (float, seconds).
    *   **Output:** A "Packet" dictionary (see below).
3.  **State Machine:**
    *   **Phase Order:** Must match the exercise mechanics.
        *   *Down-First (Squat, Bench):* `IDLE` -> `ECCENTRIC` -> `BOTTOM` -> `CONCENTRIC` -> `COMPLETE`.
        *   *Up-First (Deadlift, Row):* `IDLE` -> `CONCENTRIC` -> `TOP` -> `ECCENTRIC` -> `COMPLETE`.
4.  **Scoring:**
    *   Start at `100`. Deduct points for specific faults (e.g., `-10` for Heel Lift).
    *   Faults should be logged in a list.
5.  **Packet Structure (Return Value):**
    ```python
    {
        "state": str,              # Current state name
        "reps": int,               # Successful rep count
        "score": int,              # Current rep score (0-100)
        "feedback": str,           # Short UI message (e.g., "Knees Out!")
        "faults": List[str],       # List of fault codes detected
        "coords": List[Landmark],  # Normalized landmarks
        "raw_coords": List[Landmark], # Raw landmarks (for Visualizer)
        "metrics": dict            # Optional: Extra data like angles/velocities
    }
    ```

### D. Visualizer (`visualizer.py`)

**Purpose:**
Pure rendering. It accepts the Packet and draws the analysis on the video frame.

**Requirements:**
1.  **Method:** `draw(self, frame, packet)`
    *   **Input:** OpenCV image array (`frame`), Packet dict from Rep Logic.
    *   **Output:** Modified `frame`.
2.  **Standards:**
    *   **Colors:**
        *   Good/Pass: Green `(0, 255, 0)`
        *   Warning: Orange `(0, 165, 255)`
        *   Bad/Fail: Red `(0, 0, 255)`
        *   Text: White `(255, 255, 255)` or Black `(0, 0, 0)` background.
    *   **Elements:**
        *   **Skeleton:** Draw lines connecting joints using `raw_coords`. Highlight faulty joints in Red.
        *   **HUD:** Display Reps, Score, and State at the top.
        *   **Toast:** Display `feedback` message in the center or conspicuous location.
    *   **Robustness:** Handle `packet=None` (e.g., if user is lost) by returning original frame or a "Searching..." overlay.

---

## 3. Common Logic & Helpers

When implementing logic, consider these physics/biomechanics principles:

1.  **Velocity:** DO NOT assume a fixed FPS (e.g., `1/30`). Use `time.time()` or the frame timestamp passed to the process method to calculate `dt`.
    *   Formula: `velocity = (current_pos - prev_pos) / (current_time - prev_time)`.
2.  **Angles:** Use Vector Dot Product or `arctan2` to calculate joint angles.
    *   *Tip:* Always visualize your angles during debug to ensure you are measuring the acute/obtuse angle you expect.
3.  **Hysteresis:** Use buffers or thresholds when switching states to avoid flickering (e.g., "Must be at bottom for 3 frames").
4.  **Z-Axis:** MediaPipe estimates depth (Z). Use it for "Knees caving in" (Valgus) checks or balance checks, but be aware it is less accurate than X/Y.

## 4. Factory Registration

Once a new pipeline is created, it must be registered in `pipelines.py` -> `PipelineFactory.EXERCISE_MAP`.

```python
'new_exercise': {
    'module': 'new-exercise-folder',
    'classes': {
        'gatekeeper': ('gatekeeper.gatekeeper', 'Gatekeeper'),
        'normalizer': ('normalizer.normalizer', 'Normalizer'),
        'rep': ('rep.rep', 'RepLogic'),
        'visualizer': ('visualizer.visualizer', 'Visualizer')
    }
}
```
