import numpy as np

from tools.config_instance import ConfigInstance
from tools.frame_getter import FrameGetter
from tools.frame_normalizator import FrameNormalizator
from tools.gaze_model_inference import GazeModelInference
from tools.geometry_utils import GeometryUtils
from tools.gaze_vector_utils import GazeVectorUtils

config_instance = ConfigInstance()
use_normal_camera = config_instance.use_normal_camera

frame_getter = FrameGetter()
normalizator = FrameNormalizator()
gaze_inference = GazeModelInference()

getter = frame_getter.getter

while True:
    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    frame = frame_getter.get()

    if use_normal_camera:
        normalizator_success = normalizator.run_image_normalization(frame.copy())
    else:
        depth_frame = frame_getter.getter.depth_frame
        depth_camera_intrinsics = frame_getter.getter.intrinsics
        normalizator_success = normalizator.run_image_normalization_with_depth_camera(frame.copy(),depth_frame.copy(),depth_camera_intrinsics)

    if not normalizator_success:
        print("Normalizator error.")
        continue
    
    yaw, pitch = gaze_inference.inference(normalizator.normalizated_image)
    normalizated_gaze_2d = np.array([pitch,yaw])
    normalizated_gaze_3d = GazeVectorUtils.gaze2d_to_gaze3d(normalizated_gaze_2d)

    gaze_3d = GazeVectorUtils.denormalizated_3d_gaze_vector(normalizated_gaze_3d,normalizator.R_mat)
    face_center = normalizator.face_center

    print("Point in screen mm", GeometryUtils.ray_plan_intersection(face_center,gaze_3d))





