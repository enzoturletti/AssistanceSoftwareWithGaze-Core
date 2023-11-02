from tools.facial_landmarks_detector import FaceLandmarksDetector
from tools.frame_getter import FrameGetter

import cv2
import numpy as np


face_landmarks_detector = FaceLandmarksDetector()

frame_getter = FrameGetter()

while True:
    if (cv2.waitKey(1) & 0xFF == 27) or frame_getter.stopped:
        frame_getter.stop()
        break

    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    frame = frame_getter.get()
    success, landmarks = face_landmarks_detector.detect(frame)

    if not success:
        continue

    for (x,y) in landmarks:
        x = int(x)
        y = int(y)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Detection',frame)



