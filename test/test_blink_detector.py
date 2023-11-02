import cv2 
import numpy as np
import matplotlib.pyplot as plt

from tools.blink_detector import BlinkDetector
from tools.frame_getter import FrameGetter
from tools.facial_landmarks_detector import FaceLandmarksDetector

frame_getter = FrameGetter()
blink_detector = BlinkDetector()
face_landmarks_detector = FaceLandmarksDetector()

while True:
    if (cv2.waitKey(1) & 0xFF == 27) or frame_getter.stopped:
        frame_getter.stop()
        break

    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    frame = frame_getter.get().copy()
    success, facelandmarks = face_landmarks_detector.detect(frame)

    if not success:
        continue

    right_eye = blink_detector.det_right_eye(frame,facelandmarks)
    result = blink_detector.detect(frame,facelandmarks)

    cv2.putText(frame,  "Is right eye open? " + ("Yes" if result else "No")  ,(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if result else (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow('Right Eye',right_eye)
    cv2.imshow('Webcam image',frame)


