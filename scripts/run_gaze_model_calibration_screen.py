from tools.calibration_screen import CalibrationScreen
from tools.gaze_model_inference import GazeModelInference

gaze_model_inference = GazeModelInference()
calibration_screen = CalibrationScreen(gaze_model_inference)
calibration_screen.show()
