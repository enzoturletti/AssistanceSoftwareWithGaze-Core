import cv2
import numpy as np
from tools.frame_getter import FrameGetter

frame_getter = FrameGetter()

while True:
    if (cv2.waitKey(1) & 0xFF == 27) or frame_getter.stopped:
        frame_getter.stop()
        break

    if not isinstance(frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    cv2.imshow('Frame',frame_getter.get())

