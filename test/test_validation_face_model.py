import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tools.config_instance import ConfigInstance
from data.face_model import FaceModel
from tools.frame_getter import FrameGetter
from tools.facial_landmarks_detector import FaceLandmarksDetector
from tools.facial_landmarks_3d_detector import FacialLandmarks3dDetector
from tools.frame_normalizator import FrameNormalizator

config_instance = ConfigInstance()
use_normal_camera = config_instance.use_normal_camera

frame_getter = FrameGetter()

face_landmarks_detector = FaceLandmarksDetector()
face_landmarks_3d_detector = FacialLandmarks3dDetector()
frame_normalizator = FrameNormalizator()

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].set_title('Facial Landmarks Promedium X')
axs[0].set_ylabel('X Promedio')

axs[1].set_title('Facial Landmarks Promedium Y')
axs[1].set_ylabel('Y Promedio')

axs[2].set_title('Facial Landmarks Promedium Z')
axs[2].set_ylabel('Z Promedio')
axs[2].set_xlabel('t')


axs[0].scatter(0, 0, color='blue', label='Aproximado')
axs[0].scatter(0, 0, color='red', label='Medido')
axs[0].legend(loc='upper right')

time.sleep(2)

def animate(i):
    success, facial_landmarks_2d = face_landmarks_detector.detect(frame_getter.get())
    if not success:
        return False
    
    rot, tvec = frame_normalizator.calculate_rot_tvec(facial_landmarks_2d, FaceModel().get())
    facial_landmarks_3d_estimated_with_mp, face_model = face_landmarks_3d_detector.estimate_with_face_model(rot, tvec)

    facelandmarks_centers = frame_normalizator.calculate_face_center(facial_landmarks_3d_estimated_with_mp)
    x_aprox = facelandmarks_centers[0]
    y_aprox = facelandmarks_centers[1]
    z_aprox = facelandmarks_centers[2]


    if not use_normal_camera:
        new_facial_landmarks_2d, facial_landmarks_3d = face_landmarks_3d_detector.detect_with_depth_camera(frame_getter.getter.depth_frame, frame_getter.getter.intrinsics, facial_landmarks_2d)

        try:
            facelandmarks_centers_real = frame_normalizator.calculate_face_center(facial_landmarks_3d)
            x = facelandmarks_centers_real[0]
            y = facelandmarks_centers_real[1]
            z = facelandmarks_centers_real[2]

        except:
            return

    axs[0].scatter(i, x_aprox, color='blue', label='Aproximado')
    axs[0].scatter(i, x, color='red', label='Medido')

    axs[1].scatter(i, y_aprox, color='blue')
    axs[1].scatter(i, y, color='red')

    axs[2].scatter(i, z_aprox, color='blue')
    axs[2].scatter(i, z, color='red')


ani = animation.FuncAnimation(fig, animate, fargs=(), interval=10)
plt.show()






