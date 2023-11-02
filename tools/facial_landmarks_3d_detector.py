#This class use MediaPipe Model to estimate FaceModel

import numpy as np
import pyrealsense2 as rs

from data.face_model import FaceModel

class FacialLandmarks3dDetector:
    def __init__(self):
        self.face_model_3d = FaceModel().get()
        
    def estimate_with_face_model(self, rot, tvec):
        REYE_INDICES: np.ndarray = np.array([33, 133])
        LEYE_INDICES: np.ndarray = np.array([362, 263])
        MOUTH_INDICES: np.ndarray = np.array([185, 409])

        facial_landmarks_3d = self.face_model_3d @ rot.T + tvec
        face_model = facial_landmarks_3d[np.concatenate([REYE_INDICES, LEYE_INDICES, MOUTH_INDICES])]

        return facial_landmarks_3d, face_model
    
    def detect_with_depth_camera(self,depth_image, depth_camera_intrinsics,facial_landmarks_2d):
        pixels_vector = []
        depth_vector = []

        reference_nouse_1 = facial_landmarks_2d[0]
        reference_nouse_px_1 = int(reference_nouse_1[0])
        reference_nouse_py_1 = int(reference_nouse_1[1])

        reference_nouse_2 = facial_landmarks_2d[3]
        reference_nouse_px_2 = int(reference_nouse_2[0])
        reference_nouse_py_2 = int(reference_nouse_2[1])

        reference_nouse_3 = facial_landmarks_2d[4]
        reference_nouse_px_3 = int(reference_nouse_3[0])
        reference_nouse_py_3 = int(reference_nouse_3[1])
        
        reference_nouse_point_1 = rs.rs2_deproject_pixel_to_point(depth_camera_intrinsics, (reference_nouse_px_1,reference_nouse_py_1), depth_image[reference_nouse_py_1][reference_nouse_px_1])
        reference_nouse_distance_1 = reference_nouse_point_1[2]/10000

        reference_nouse_point_2 = rs.rs2_deproject_pixel_to_point(depth_camera_intrinsics, (reference_nouse_px_2,reference_nouse_py_2), depth_image[reference_nouse_py_2][reference_nouse_px_2])
        reference_nouse_distance_2 = reference_nouse_point_2[2]/10000

        reference_nouse_point_3 = rs.rs2_deproject_pixel_to_point(depth_camera_intrinsics, (reference_nouse_px_3,reference_nouse_py_3), depth_image[reference_nouse_py_3][reference_nouse_px_3])
        reference_nouse_distance_3 = reference_nouse_point_3[2]/10000

        reference_nouse_distance = reference_nouse_distance_1
        if reference_nouse_distance == 0.0:
            reference_nouse_distance = reference_nouse_distance_2
        if reference_nouse_distance == 0.0:
            reference_nouse_distance = reference_nouse_distance_3

        for i, landmark_2d in enumerate(facial_landmarks_2d):
            px = int(landmark_2d[0])
            py = int(landmark_2d[1])

            if px >= depth_image.shape[1] or py >= depth_image.shape[0] or px < 0 or py < 0:
                continue 

            pixels_vector.append((px,py))
            depth_vector.append(depth_image[py][px])

            new_facial_landmarks_2d = []
            facial_landmarks_3d = []
            for i, pixel in enumerate(pixels_vector):
                point_3d = rs.rs2_deproject_pixel_to_point(depth_camera_intrinsics, pixel, depth_vector[i])

                if point_3d[2]/10000-reference_nouse_distance > 0.3:
                    continue

                if point_3d[0] == 0 and point_3d[1] == 0:
                    continue
                
                new_facial_landmarks_2d.append(pixel)
                facial_landmarks_3d.append(point_3d)

        facial_landmarks_3d = (np.array(facial_landmarks_3d)/10000).astype(np.float32)
        new_facial_landmarks_2d = np.array(new_facial_landmarks_2d).astype(np.float32)
        
        return new_facial_landmarks_2d,facial_landmarks_3d


    