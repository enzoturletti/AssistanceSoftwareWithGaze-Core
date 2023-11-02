import cv2
import numpy as np
import collections

from tools.config_instance import ConfigInstance
from tools.frame_getter import FrameGetter
from tools.frame_normalizator import FrameNormalizator
from tools.gaze_inference import GazeInference
from tools.converter_mm_px import Converter
from tools.geometry_utils import GeometryUtils
from tools.gaze_vector_utils import GazeVectorUtils
from tools.user_calibration_engine import UserCalibrationEngine
from tools.video_demostration_builder import VideoDemostrationBuilder

class TestingPrecisionScreen:
        def __init__(self, inference_engine : GazeInference):

            # Config read.
            self.config_instance = ConfigInstance()
            self.use_normal_camera = self.config_instance.use_normal_camera
            self.number_of_circles = self.config_instance.number_of_circles
            self.res_x = self.config_instance.resolution_x
            self.res_y = self.config_instance.resolution_y
            self.use_user_gaze_calibration = self.config_instance.use_user_gaze_calibration

            #self.output_path = config_instance.calibration_gaze_model_output

            # Inference
            self.inference_engine = inference_engine
            self.frame_normalizator = FrameNormalizator()
            self.frame_getter = FrameGetter()            
            self.converter = Converter()

            # Screen
            self.calibration_index_x = 0
            self.calibration_index_y = 0
            self.data = []
            self.px_buffer = collections.deque(maxlen=7)
            self.py_buffer = collections.deque(maxlen=7)
            self.init_interval_x = np.linspace(0, self.res_x,self.number_of_circles,endpoint=False)
            self.init_interval_y = np.linspace(0, self.res_y,self.number_of_circles,endpoint=False)
            self.white_screen = np.zeros([self.res_y,self.res_x,3],dtype=np.uint8)
            self.white_screen.fill(0)
            self.draw_circles(self.init_interval_x,self.init_interval_y,(255,0,0))

            # User Calibration
            if self.use_user_gaze_calibration:
                self.user_calibration_engine = UserCalibrationEngine()

            # Video Demo
            if self.config_instance.video_demo_activated: self.video_demo_build = VideoDemostrationBuilder("testing_precision_screen")


        def show(self):
            while True:
                if (cv2.waitKey(1) & 0xFF == 27) or self.frame_getter.stopped:
                    if self.config_instance.video_demo_activated: self.video_demo_build.finish()
                    self.frame_getter.stop()
                    break

                if not isinstance(self.frame_getter.get(),np.ndarray):
                    print("Skip")
                    continue
                
                self.frame = self.frame_getter.get()

                # Normalization
                if self.use_normal_camera:
                    normalizator_success = self.frame_normalizator.run_image_normalization(self.frame.copy())
                else:
                    depth_frame = self.frame_getter.getter.depth_frame
                    depth_camera_intrinsics = self.frame_getter.getter.intrinsics
                    normalizator_success = self.frame_normalizator.run_image_normalization_with_depth_camera(self.frame.copy(),depth_frame.copy(),depth_camera_intrinsics)

                if not normalizator_success:
                    print("Normalizator error.")
                    continue

                face_center = self.frame_normalizator.face_center
                R_mat = self.frame_normalizator.R_mat
                
                # Inference
                yaw, pitch = self.inference_engine.inference(self.frame_normalizator.normalizated_image)
                normalizated_gaze_2d = np.array([yaw,pitch])
                normalizated_gaze_3d = GazeVectorUtils.gaze2d_to_gaze3d(normalizated_gaze_2d)
                gaze_3d = GazeVectorUtils.denormalizated_3d_gaze_vector(normalizated_gaze_3d,R_mat)

                # Ray Plan Intersection
                x_cam_coordinate_mm, y_cam_coordinate_mm, z_cam_coordinate_mm = GeometryUtils.ray_plan_intersection(face_center,gaze_3d)

                # Convertion to pixels
                px, py = self.converter.get_pixels(x_cam_coordinate_mm,y_cam_coordinate_mm)

                # User calibration engine
                if self.use_user_gaze_calibration:
                    px, py = self.user_calibration_engine.correct(px,py)
                
                self.px_buffer.append(px)
                self.py_buffer.append(py)
                self.px = np.asarray(self.px_buffer).mean(axis=0)
                self.py = np.asarray(self.py_buffer).mean(axis=0)
                
                tempImg = self.white_screen.copy()
                cv2.circle(tempImg, (int(self.px),int(self.py)) , radius=5, color=(255, 255, 255), thickness=-1)

                if self.config_instance.video_demo_activated: self.video_demo_build.record(tempImg,self.frame)

                cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                #cv2.setMouseCallback('frame',self.click_callback)
                cv2.imshow('frame', tempImg)

        def draw_circles(self,centers_in_x, centers_in_y,circle_color):
            for center_x in centers_in_x:
                for center_y in centers_in_y:
                    self.draw_circle(center_x,center_y,circle_color)

        def draw_circle(self,center_x, center_y,circle_color):
            self.actual_x_center = int(center_x+self.res_x/(self.number_of_circles*2))
            self.actual_y_center = int(center_y+self.res_y/(self.number_of_circles*2))
            self.white_screen = cv2.circle(self.white_screen, (self.actual_x_center,self.actual_y_center), radius=25, color=circle_color, thickness=-1)

        def draw_estimation(self):
            self.white_screen = cv2.circle(self.white_screen, (int(self.px),int(self.py)), radius=5, color=(255,0,255), thickness=-1)

        #def click_callback(self,event,x,y,flags,param):
        #    if event == cv2.EVENT_LBUTTONDOWN and len(self.data) < self.number_of_circles*self.number_of_circles:
        #        self.draw_circle(self.init_interval_x[self.calibration_index_x],self.init_interval_y[self.calibration_index_y],(0,255,0))
        #        self.collect_data()
        #        self.draw_estimation()
        #        self.calibration_index_x = self.calibration_index_x + 1
        #
        #        if self.calibration_index_x == self.number_of_circles and self.calibration_index_y != self.number_of_circles:
        #            self.calibration_index_x = 0
        #            self.calibration_index_y = self.calibration_index_y + 1
        #    else:
        #        self.data =  list(filter(None, self.data))
        #        with open(self.output_path, 'w') as fp:
        #            for item in self.data:
        #                fp.write("%s\n" % item)

        def collect_data(self):
            px = self.actual_x_center
            py = self.actual_y_center
            item = {
                "px_estimated" :  int(self.px),
                "py_estimated" :  int(self.py),
                'px' : px,
                'py' : py
                }
            self.data.append(item)




