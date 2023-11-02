from threading import Thread
from tools.gaze_inference import GazeInference
from tools.frame_getter import FrameGetter
from tools.converter_mm_px import Converter
from tools.frame_normalizator import FrameNormalizator
from tools.gaze_model_inference import GazeModelInference
from tools.gaze_vector_utils import GazeVectorUtils
from tools.geometry_utils import GeometryUtils
from tools.config_instance import ConfigInstance
from tools.blink_detector import BlinkDetector
from tools.user_calibration_engine import UserCalibrationEngine

import collections
import numpy as np

class PipelineGetter:
    def __init__(self):
        self.stopped = False
        self.ok = True

        #Config
        config_instance = ConfigInstance()
        self.use_normal_camera = config_instance.use_normal_camera
        res_x = config_instance.resolution_x
        res_y = config_instance.resolution_y
        self.use_user_gaze_calibration = config_instance.use_user_gaze_calibration

        # Instances
        self.frame_getter = FrameGetter()
        self.normalizator = FrameNormalizator()
        self.gaze_inference = GazeModelInference()
        self.converter = Converter()
        self.blink_detector = BlinkDetector()

        if self.use_user_gaze_calibration:
            self.user_calibration_engine = UserCalibrationEngine()

        # Buffer
        self.px_buffer = collections.deque(maxlen=7)
        self.py_buffer = collections.deque(maxlen=7)
        self.px_buffer.append(res_x/2)
        self.py_buffer.append(res_y/2)

        # Outputs initialization
        self.px = np.asarray(self.px_buffer).mean(axis=0)
        self.py = np.asarray(self.py_buffer).mean(axis=0)
        self.eye_open = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def stop(self):
        self.stopped = True
        self.frame_getter.stop()

    def get(self):
        while not self.stopped or self.frame_getter.stopped:
            if not self.ok:
                self.stop()
            else:
                self.frame = self.frame_getter.get().copy()
                
                # Normalization
                if self.use_normal_camera:
                    normalizator_success = self.normalizator.run_image_normalization(self.frame)
                else:
                    depth_frame = self.frame_getter.getter.depth_frame.copy()
                    depth_camera_intrinsics = self.frame_getter.getter.intrinsics
                    normalizator_success = self.normalizator.run_image_normalization_with_depth_camera(self.frame,depth_frame,depth_camera_intrinsics)

                if not normalizator_success:
                    print("Normalizator error.")
                    continue

                # Blink detector
                self.eye_open = self.blink_detector.detect(self.frame,self.normalizator.facial_landmarks_2d)

                if not self.eye_open:
                    continue

                face_center = self.normalizator.face_center
                R_mat = self.normalizator.R_mat

                # Inference
                yaw, pitch = self.gaze_inference.inference(self.normalizator.normalizated_image)
                normalizated_gaze_2d = np.array([yaw,pitch])
                normalizated_gaze_3d = GazeVectorUtils.gaze2d_to_gaze3d(normalizated_gaze_2d)
                gaze_3d = GazeVectorUtils.denormalizated_3d_gaze_vector(normalizated_gaze_3d,R_mat)

                # Ray Plan Intersection
                x_cam_coordinate_mm, y_cam_coordinate_mm, z_cam_coordinate_mm = GeometryUtils.ray_plan_intersection(face_center,gaze_3d)

                # Convertion to pixels
                px, py = self.converter.get_pixels(x_cam_coordinate_mm,y_cam_coordinate_mm)

                if self.use_user_gaze_calibration:
                    px, py = self.user_calibration_engine.correct(px,py)
                
                self.px_buffer.append(px)
                self.py_buffer.append(py)
                self.px = np.asarray(self.px_buffer).mean(axis=0)
                self.py = np.asarray(self.py_buffer).mean(axis=0)