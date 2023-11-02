
from tools.depth_camera_frame_getter import DepthCameraFrameGetter
import cv2
import numpy as np

depth_camera_frame_getter = DepthCameraFrameGetter().start()
while True:
    if (cv2.waitKey(1) & 0xFF == 27) or depth_camera_frame_getter.stopped:
        depth_camera_frame_getter.stop()
        break

    if not isinstance(depth_camera_frame_getter.frame,np.ndarray) or not isinstance(depth_camera_frame_getter.depth_frame,np.ndarray):
        print("Skip")
        continue

    cv2.imshow('Frame',depth_camera_frame_getter.frame)
    cv2.imshow('Depth Frame',depth_camera_frame_getter.depth_frame)

