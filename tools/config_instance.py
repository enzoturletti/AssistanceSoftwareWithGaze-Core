import configparser
import math
class ConfigInstance:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance
    
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Demo
        self.video_demo_activated = int(config.get('demo', 'allow_video_demostration'))

        # Cameras
        self.use_normal_camera = int(config.get('camera', 'use_normal_camera'))
        self.normalizated_distance = float(config.get("normalizated_camera","normalizated_distance"))
        self.camera_calibration = config.get('paths', 'camera_calibration')
        
        # Paths
        self.camera_intrinsic_path = config.get('paths', 'camera_intrinsinc')
        self.normalizated_camera_intrinsic_path = config.get('paths', 'normalized_camera_intrinsic')
        self.face_model_path = config.get('paths', 'face_model')
        self.gaze_model_path = config.get('paths', 'gaze_model')

        # Screen Size
        self.resolution_x = int(config.get('screen_size', 'resolution_x'))
        self.resolution_y = int(config.get('screen_size', 'resolution_y'))
        self.offset_camera = float(config.get('screen_size', 'offset_camera'))
        self.aspect_ratio_horizontal = int(config.get('screen_size', 'aspect_ratio_horizontal'))
        self.aspect_ratio_vertical = int(config.get('screen_size', 'aspect_ratio_vertical'))
        self.diagonal_size = float(config.get('screen_size', 'diagonal_size'))
        
        aspect_ratio = self.aspect_ratio_horizontal/self.aspect_ratio_vertical
        self.height_screen = float(math.sqrt((self.diagonal_size ** 2) / (1 + aspect_ratio ** 2)) * 25.4)
        self.width_screen = float(aspect_ratio * self.height_screen)

        # Calibration Screen
        self.resting_time = int(config.get('calibration_screen', 'resting_time'))
        self.number_of_circles = int(config.get('calibration_screen', 'number_of_circles'))
        self.calibration_gaze_model_output = config.get('calibration_screen', 'calibration_gaze_model_output')
        self.use_user_gaze_calibration = int(config.get('calibration_screen', 'use_user_gaze_calibration'))

        # Click calibration
        self.click_calibration_path = config.get('click_calibration', 'click_calibration_path')
        self.reference_open_eye_1 = config.get('click_calibration', 'reference_open_eye_1')
        self.reference_open_eye_2 = config.get('click_calibration', 'reference_open_eye_2')
        self.reference_open_eye_3 = config.get('click_calibration', 'reference_open_eye_3')
        self.reference_open_eye_4 = config.get('click_calibration', 'reference_open_eye_4')
        self.reference_open_eye_5 = config.get('click_calibration', 'reference_open_eye_5')
        self.reference_close_eye =  config.get('click_calibration', 'reference_close_eye')
        
        # Grid Screen
        self.grid_size = int(config.get('grid_screen', 'grid_size'))


