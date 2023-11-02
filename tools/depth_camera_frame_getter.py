import cv2
import pyrealsense2 as rs
import numpy as np
from threading import Thread


class DepthCameraFrameGetter:
    def __init__(self):
        self.ok = True
        self.stopped = False

        # Configuración de la cámara
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Iniciar la captura de imágenes
        self.pipeline.start(config)

        # Obtener la matriz de calibración de la cámara
        profile = self.pipeline.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.intrinsics = depth_profile.get_intrinsics()

        self.frame = None
        self.depth_frame = None

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def stop(self):
        self.pipeline.stop()
        self.stopped = True

    def get(self):
        while not self.stopped:
            if not self.ok:
                self.stop()
            else:

                # Obtener los datos de profundidad y textura de la cámara.
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convierte a array de numpy.
                self.frame = np.asanyarray(color_frame.get_data())
                self.depth_frame = np.asanyarray(depth_frame.get_data())

