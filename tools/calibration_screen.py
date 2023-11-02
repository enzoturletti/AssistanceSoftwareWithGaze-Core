import cv2
import numpy as np
import collections
import time

from tools.config_instance import ConfigInstance
from tools.frame_getter import FrameGetter
from tools.frame_normalizator import FrameNormalizator
from tools.gaze_inference import GazeInference
from tools.gaze_model_inference import GazeModelInference
from tools.converter_mm_px import Converter
from tools.geometry_utils import GeometryUtils
from tools.gaze_vector_utils import GazeVectorUtils
from tools.video_demostration_builder import VideoDemostrationBuilder
from tools.video_demostration_builder import DemoType


class CalibrationScreen:
        def __init__(self, inference_engine : GazeInference):

            # Config read.

            config_instance = ConfigInstance()
            self.use_normal_camera = config_instance.use_normal_camera
            self.res_x = config_instance.resolution_x
            self.res_y = config_instance.resolution_y
            self.circles_number = config_instance.number_of_circles
            self.resting_time = config_instance.resting_time
            self.output_path = config_instance.calibration_gaze_model_output
            self.video_demo_activated = config_instance.video_demo_activated

            # Inference
            self.inference_engine = inference_engine
            self.frame_normalizator = FrameNormalizator()
            self.frame_getter = FrameGetter()            
            self.converter = Converter()

            self.calibration_index_x = 0
            self.calibration_index_y = 0
            self.data = []
            self.px_buffer = collections.deque(maxlen=7)
            self.py_buffer = collections.deque(maxlen=7)

            # Video demo
            if self.video_demo_activated: self.video_demo_builder = VideoDemostrationBuilder("calibration_screen",DemoType.CALIBRATION_SCREEN)

            # UI
            self.circles_center_x = np.linspace(0, self.res_x,self.circles_number,endpoint=True)
            self.circles_center_y = np.linspace(0, self.res_y,self.circles_number,endpoint=False)

            self.finish = False
            self.resting = True
            self.start_time = time.time()

            self.actual_row = 0
            self.actual_x_center = int(self.circles_center_x[0])
            self.actual_y_center = int(self.circles_center_y[0]+self.res_y/(self.circles_number*2))
           
            self.screen = np.zeros([self.res_y,self.res_x,3],dtype=np.uint8)
            self.screen.fill(1)

            for i in range(self.circles_number):
                auxiliar_y = int(self.circles_center_y[i]+self.res_y/(self.circles_number*2))
                self.screen = cv2.circle(self.screen, (self.actual_x_center,auxiliar_y), radius=5, color=(255,0,255), thickness=-1)

            self.screen_backup = self.screen.copy()

        def show(self):
            while True:
                if self.finish:
                    with open(self.output_path, 'w') as fp:
                        for item in self.data:
                            fp.write("%s\n" % item)
                    self.frame_getter.stop()
                    if self.video_demo_activated: self.video_demo_builder.finish()
                    break

                if (cv2.waitKey(1) & 0xFF == 27):
                    self.frame_getter.stop()
                    if self.video_demo_activated: self.video_demo_builder.finish()
                    break

                if not isinstance(self.frame_getter.get(),np.ndarray):
                    print("Skip")
                    continue
                
                if not self.finish:
                    self.animate()

                self.frame = self.frame_getter.get()

                # Normalization
                if self.use_normal_camera:
                    normalizator_success = self.frame_normalizator.run_image_normalization(self.frame.copy())
                else:
                    depth_frame = self.frame_getter.getter.depth_frame
                    depth_camera_intrinsics = self.frame_getter.getter.intrinsics
                    normalizator_success = self.frame_normalizator.run_image_normalization_with_depth_camera(self.frame.copy(),depth_frame.copy(),depth_camera_intrinsics)

                if not normalizator_success:
                    
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
                
                self.px_buffer.append(px)
                self.py_buffer.append(py)
                self.px = np.asarray(self.px_buffer).mean(axis=0)
                self.py = np.asarray(self.py_buffer).mean(axis=0)

                # Collect data
                if not self.resting:
                    item = {
                    "px_estimated" :  int(self.px),
                    "py_estimated" :  int(self.py),
                    'px' : self.actual_x_center,
                    'py' : self.actual_y_center
                    }
                    self.data.append(item)

                temp_screen = self.screen.copy()
                cv2.circle(temp_screen, (int(self.px),int(self.py)) , radius=5, color=(255, 255, 255), thickness=-1)

                if self.video_demo_activated: self.video_demo_builder.record(temp_screen,self.frame)

                cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow('frame', temp_screen)


        def animate(self):
            if self.resting:
                time_elapsed = time.time() - self.start_time

                seconds_left = self.resting_time-time_elapsed

                if seconds_left < 0:
                    seconds_left = 0

                self.rest_popup(int(seconds_left))

                if seconds_left == 0:
                    self.resting = False
                    self.screen = self.screen_backup.copy()

                return

            if self.actual_x_center < self.circles_center_x[len(self.circles_center_x)-1]+self.res_x/(self.circles_number*2):
                self.actual_x_center += 15

                self.screen = cv2.circle(self.screen, (self.actual_x_center,self.actual_y_center), radius=25, color=(255,0,0), thickness=-1)
                self.screen = cv2.circle(self.screen, (self.actual_x_center,self.actual_y_center), radius=5, color=(255,0,255), thickness=-1)
            else:
                self.actual_row += 1

                if self.actual_row == self.circles_number:
                    self.finish = True
                    return

                self.resting = True
                self.screen_backup = self.screen.copy()
                self.start_time = time.time()

                self.actual_x_center = int(self.circles_center_x[0])
                self.actual_y_center = int(self.circles_center_y[self.actual_row]+self.res_y/(self.circles_number*2))


        def rest_popup(self, seconds_left):     
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 2
            text = f"{seconds_left}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            x = (self.res_x - text_size[0]) // 2
            y = (self.res_y + text_size[1]) // 2

            self.screen = self.screen_backup.copy()
            cv2.putText(self.screen, text, (x, y), font, font_scale, (255, 255, 255), font_thickness)
   




