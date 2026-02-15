import cv2
import mediapipe as mp
import importlib

class Landmark:
    """Wrapper for normalized output."""
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class ExercisePipeline:
    def process(self, frame, landmarks, timestamp=None):
        raise NotImplementedError

class GenericPipeline(ExercisePipeline):
    def __init__(self, exercise_name, modules):
        self.exercise_name = exercise_name
        # Instantiate the classes dynamically
        # modules is a dict: {'gatekeeper': Class, 'normalizer': Class, 'rep': Class, 'visualizer': Class}
        self.gatekeeper = modules['gatekeeper']()
        self.normalizer = modules['normalizer']()
        self.rep_class = modules['rep']
        self.visualizer = modules['visualizer']()
        
        self.rep_logic = None
        self.calibration_data = None
        self.status_message = "Initializing..."

    def process(self, frame, landmarks, timestamp=None):
        if not landmarks:
            return frame

        # 1. Gatekeeper (Calibration Phase)
        if not self.rep_logic:
            passed, msg, cal_data = self.gatekeeper.check(landmarks)
            self.status_message = msg
            
            # Draw Gatekeeper Overlay
            cv2.putText(frame, f"SETUP: {msg}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            if passed:
                self.calibration_data = cal_data
                # Instantiate Rep Logic with Calibration Data
                self.rep_logic = self.rep_class(cal_data)
                print(f"{self.exercise_name} Calibration Complete: {cal_data}")
            
            return frame

        # 2. Normalization
        # Pass calibration data (if supported) to Normalizer
        normalized_landmarks = self.normalizer.process(landmarks, self.calibration_data)

        # 3. Rep Logic
        # Pass both normalized and raw landmarks
        packet = self.rep_logic.process(normalized_landmarks, raw_landmarks=landmarks, timestamp=timestamp)

        # 4. Visualization
        return self.visualizer.draw(frame, packet)

class PipelineFactory:
    # Map friendly names to module paths and class names
    # Structure: 'key': {'path': 'module_path', 'classes': {'gatekeeper': 'ClassName', ...}}
    EXERCISE_MAP = {
        'squat': {
            'module': 'squat',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper-squat', 'Gatekeeper'),
                'normalizer': ('normalizer.normalizer', 'SquatNormalizer'),
                'rep': ('rep.squat_rep', 'SquatRep'),
                'visualizer': ('visualizer.visualizer', 'Visualizer') # Assuming standard name? Check.
            }
        },
        'deadlift': {
            'module': 'deadlift',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper-deadlift', 'DeadliftGatekeeper'),
                'normalizer': ('normalizer.normalizer', 'DeadliftNormalizer'),
                'rep': ('rep.rep', 'DeadliftRep'),
                'visualizer': ('visualizer.visualizer', 'DeadliftVisualizer') # Check name
            }
        },
        'bench': {
            'module': 'flat_barbell_press',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'BenchGatekeeper'),
                'normalizer': ('normalizer.normalizer', 'BenchNormalizer'),
                'rep': ('rep.rep', 'BenchPressRep'),
                'visualizer': ('visualizer.visualizer', 'BenchVisualizer')
            }
        },
        'seated_row': {
            'module': 'seated-row',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'Gatekeeper'),
                'normalizer': ('normalizer.normalizer', 'SeatedRowNormalizer'),
                'rep': ('rep.rep', 'SeatedRowRep'),
                'visualizer': ('visualizer.visualizer', 'Visualizer') # Check name
            }
        },
        'tricep_pushdown': {
            'module': 'tricep-pushdowns',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'TricepPushdownGatekeeper'),
                'normalizer': ('normalizer.normalizer', 'TricepPushdownNormalizer'),
                'rep': ('rep.rep', 'TricepPushdownRep'),
                'visualizer': ('visualizer.visualizer', 'TricepPushdownVisualizer')
            }
        },
        'lat_pulldown': {
            'module': 'lat-pulldowns',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'Gatekeeper'),
                'normalizer': ('normalizer.normalizer', 'Normalizer'), # Check name
                'rep': ('rep.rep', 'LatPullRep'),
                'visualizer': ('visualizer.visualizer', 'Visualizer') # Check name
            }
        },
        'barbell_row': {
            'module': 'barbell-rows',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'Gatekeeper'),
                'normalizer': ('normalizer.normalizer', 'BarbellRowNormalizer'), # Check name
                'rep': ('rep.rep', 'BarbellRowRep'), # Check name
                'visualizer': ('visualizer.visualizer', 'Visualizer') # Check name
            }
        },
        'hamstring_curl': {
            'module': 'hamstring-curls',
            'classes': {
                'gatekeeper': ('gatekeeper.gatekeeper', 'Gatekeeper'),
                'normalizer': ('normalizer.normalizer', 'Normalizer'), # Check name
                'rep': ('rep.rep', 'RepLogic'), # Check name
                'visualizer': ('visualizer.visualizer', 'Visualizer') # Check name
            }
        }
    }

    @staticmethod
    def get_pipeline(exercise_name):
        # Normalize name
        key = exercise_name.lower().replace(' ', '_').replace('-', '_')
        
        # Aliases
        if key in ['bench_press', 'flat_barbell_press']: key = 'bench'
        if key in ['row', 'seated_rows']: key = 'seated_row'
        if key in ['pushdown', 'tricep_pushdowns']: key = 'tricep_pushdown'
        if key in ['deadlifts']: key = 'deadlift'
        if key in ['squats']: key = 'squat'

        if key not in PipelineFactory.EXERCISE_MAP:
            raise ValueError(f"Unknown exercise: {exercise_name}")

        config = PipelineFactory.EXERCISE_MAP[key]
        module_base = config['module']
        
        loaded_modules = {}
        
        # Dynamically import classes
        for component, (sub_path, class_name) in config['classes'].items():
            full_module_path = f"{module_base}.{sub_path}"
            try:
                mod = importlib.import_module(full_module_path)
                cls = getattr(mod, class_name)
                loaded_modules[component] = cls
            except (ImportError, AttributeError) as e:
                # Fallback for common naming conventions if explicit map fails
                # (This is a safety net for "Visualizer" vs "SquatVisualizer" ambiguity)
                raise ImportError(f"Failed to load {component} for {key}: {e}")

        return GenericPipeline(key, loaded_modules)