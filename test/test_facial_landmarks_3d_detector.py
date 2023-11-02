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

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.set_title('Surface plot')

time.sleep(2)

def animate(i):
    success, facial_landmarks_2d = face_landmarks_detector.detect(frame_getter.get())
    if not success:
        return False
    
    ax.clear()
    ax.set_xlim(-0.4,0.4)
    ax.set_ylim(-0.1,0.2)
    ax.set_zlim(-0.1, 1)

    rot, tvec = frame_normalizator.calculate_rot_tvec(facial_landmarks_2d, FaceModel().get())
    facial_landmarks_3d_estimated_with_mp, face_model = face_landmarks_3d_detector.estimate_with_face_model(rot, tvec)
    x__ = facial_landmarks_3d_estimated_with_mp[:,0]
    y__ = facial_landmarks_3d_estimated_with_mp[:,1]
    z__ = facial_landmarks_3d_estimated_with_mp[:,2]
        
    facelandmarks_centers = frame_normalizator.calculate_face_center(facial_landmarks_3d_estimated_with_mp)
    model_center = frame_normalizator.calculate_face_center(face_model)

    ax.plot_trisurf(x__, y__, z__, linewidth=0, antialiased=True,alpha=.5)
    ax.scatter(model_center[0],model_center[1],model_center[2])
    ax.scatter(facelandmarks_centers[0],facelandmarks_centers[1],facelandmarks_centers[2])

    if not use_normal_camera:
        new_facial_landmarks_2d, facial_landmarks_3d = face_landmarks_3d_detector.detect_with_depth_camera(frame_getter.getter.depth_frame, frame_getter.getter.intrinsics, facial_landmarks_2d)

        try:
            x_ = facial_landmarks_3d[:,0]
            y_ = facial_landmarks_3d[:,1]
            z_ = facial_landmarks_3d[:,2]

            face_center_2 = frame_normalizator.calculate_face_center(facial_landmarks_3d)
            ax.scatter(face_center_2[0],face_center_2[1],face_center_2[2])
            ax.plot_trisurf(x_, y_, z_, linewidth=0, antialiased=True,alpha=.5)
        except:
            return

face_model_3d = FaceModel.get()
x_ = face_model_3d[:, 0]
y_ = face_model_3d[:, 1]
z_ = face_model_3d[:, 2]

ani = animation.FuncAnimation(fig, animate, fargs=(), interval=50)
plt.show()





