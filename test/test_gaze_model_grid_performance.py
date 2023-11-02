import collections
import numpy as np
import cv2
from tools.config_instance import ConfigInstance
from tools.grid_screen import GridDisplay
from tools.gaze_model_inference import GazeModelInference
from tools.frame_normalizator import FrameNormalizator
from tools.frame_getter import FrameGetter
from tools.converter_mm_px import Converter
from tools.gaze_vector_utils import GazeVectorUtils
from tools.geometry_utils import GeometryUtils
from tools.user_calibration_engine import UserCalibrationEngine
from tools.video_demostration_builder import VideoDemostrationBuilder
from tools.video_demostration_builder import DemoType

# Config read.
config_instance = ConfigInstance()
use_normal_camera = config_instance.use_normal_camera
video_demo_activated = config_instance.video_demo_activated

# Inference and Normalization

inference_engine = GazeModelInference()
frame_normalizator = FrameNormalizator()
frame_getter = FrameGetter()            
converter = Converter()

if config_instance.use_user_gaze_calibration:
    user_calibration_engine = UserCalibrationEngine()

px_buffer = collections.deque(maxlen=7)
py_buffer = collections.deque(maxlen=7)

# Grid

grid_display = GridDisplay()

# Video demo
if config_instance.video_demo_activated: video_demo_build = VideoDemostrationBuilder("test_gaze_model_grid_performance",DemoType.GRID_TESTING)

while True:

    if (cv2.waitKey(1) & 0xFF == 27) or frame_getter.stopped:
        frame_getter.stop()
        if config_instance.video_demo_activated: video_demo_build.finish()
        break

    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    frame = frame_getter.get()

    # Normalization
    if use_normal_camera:
        normalizator_success = frame_normalizator.run_image_normalization(frame.copy())
    else:
        depth_frame = frame_getter.getter.depth_frame
        depth_camera_intrinsics = frame_getter.getter.intrinsics
        normalizator_success = frame_normalizator.run_image_normalization_with_depth_camera(frame.copy(),depth_frame.copy(),depth_camera_intrinsics)

    if not normalizator_success:
        print("Normalizator error.")
        continue

    face_center = frame_normalizator.face_center
    R_mat = frame_normalizator.R_mat
    
    # Inference
    yaw, pitch = inference_engine.inference(frame_normalizator.normalizated_image)
    normalizated_gaze_2d = np.array([yaw,pitch])
    normalizated_gaze_3d = GazeVectorUtils.gaze2d_to_gaze3d(normalizated_gaze_2d)
    gaze_3d = GazeVectorUtils.denormalizated_3d_gaze_vector(normalizated_gaze_3d,R_mat)

    # Ray Plan Intersection
    x_cam_coordinate_mm, y_cam_coordinate_mm, z_cam_coordinate_mm = GeometryUtils.ray_plan_intersection(face_center,gaze_3d)

    # Convertion to pixels
    px, py = converter.get_pixels(x_cam_coordinate_mm,y_cam_coordinate_mm)

    if config_instance.use_user_gaze_calibration:
        px, py = user_calibration_engine.correct(px,py)

    px_buffer.append(px)
    py_buffer.append(py)
    px = np.asarray(px_buffer).mean(axis=0)
    py = np.asarray(py_buffer).mean(axis=0)
    
    grid_display.color_grid_element(int(px),int(py))
    grid = grid_display.get_grid()

    if config_instance.video_demo_activated: video_demo_build.record(grid,frame)

    cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Grid",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Grid", grid)
