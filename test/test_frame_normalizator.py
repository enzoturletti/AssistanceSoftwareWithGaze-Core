import cv2 
import numpy as np

from tools.config_instance import ConfigInstance
from tools.frame_normalizator import FrameNormalizator
from tools.frame_getter import FrameGetter
from tools.facial_landmarks_detector import FaceLandmarksDetector

config_instance = ConfigInstance()
use_normal_camera = config_instance.use_normal_camera

frame_getter = FrameGetter()
normalizator = FrameNormalizator()
face_landmarks_detector = FaceLandmarksDetector()

while True:
    if (cv2.waitKey(1) & 0xFF == 27) or frame_getter.stopped:
        frame_getter.stop()
        break

    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    frame = frame_getter.get()

    success, landmarks_2d = face_landmarks_detector.detect(frame)

    if not success:
        continue

    if use_normal_camera:
        normalizator_success = normalizator.run_image_normalization(frame.copy())
    else:
        depth_frame = frame_getter.getter.depth_frame
        depth_camera_intrinsics = frame_getter.getter.intrinsics
        normalizator_success = normalizator.run_image_normalization_with_depth_camera(frame.copy(),depth_frame.copy(),depth_camera_intrinsics)

    if not normalizator_success:
        
        continue

    original_image = frame
    normalizated_image = normalizator.normalizated_image

    cv2.imshow('Webcam Image',original_image)
    cv2.imshow('Normalizated Image',normalizated_image)
