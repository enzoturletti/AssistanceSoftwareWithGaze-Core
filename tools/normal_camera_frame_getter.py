from threading import Thread
import cv2

class NormalCameraFrameGetter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.ok, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def stop(self):
        self.stopped = True

    def get(self):
        while not self.stopped:
            if not self.ok:
                self.stop()
            else:
                self.ok, self.frame = self.cap.read()