
from tools.normal_camera_frame_getter import NormalCameraFrameGetter
import cv2

normal_camera_frame_getter = NormalCameraFrameGetter().start()
while True:
    if (cv2.waitKey(1) & 0xFF == 27) or normal_camera_frame_getter.stopped:
        normal_camera_frame_getter.stop()
        break

    cv2.imshow('Image',normal_camera_frame_getter.frame)


