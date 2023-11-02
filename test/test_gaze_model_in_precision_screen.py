from tools.testing_precision_screen import TestingPrecisionScreen
from tools.gaze_model_inference import GazeModelInference

gaze_model_inference = GazeModelInference()
testing_screen = TestingPrecisionScreen(gaze_model_inference)
testing_screen.show()
