import numpy as np

from tools.config_instance import ConfigInstance
from tools.frame_getter import FrameGetter
from tools.frame_normalizator import FrameNormalizator
from tools.gaze_model_inference import GazeModelInference

config_instance = ConfigInstance()
use_normal_camera = config_instance.use_normal_camera

frame_getter = FrameGetter()
getter = frame_getter.getter

normalizator = FrameNormalizator()
gaze_inference = GazeModelInference()

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
        
        continue

    yaw, pitch = gaze_inference.inference(normalizator.normalizated_image)

    print("Pitch:", pitch)
    print("Yaw:", yaw)





