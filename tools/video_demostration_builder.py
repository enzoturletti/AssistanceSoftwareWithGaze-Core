import random
import time
import numpy as np
import cv2
from tools.config_instance import ConfigInstance
from enum import Enum

class DemoType(Enum):
    NORMAL = 0
    GRID_TESTING = 1
    CALIBRATION_SCREEN = 2

class VideoDemostrationBuilder:
    def __init__(self,filename,demo_type = DemoType.NORMAL, fps = 10):
        config_instance = ConfigInstance()
        self.res_x = config_instance.resolution_x
        self.res_y = config_instance.resolution_y
        self.using_user_calibration = config_instance.use_user_gaze_calibration
        self.grid_size = config_instance.grid_size
        self.img_array = []
        self.filename = filename
        self.demo_type = demo_type
        self.fps = fps
        self.last_record = ""

        # Random number generation
        self.random_number = ""
        self.time_start = 0


    def generate_random_number(self):
        if self.random_number == "":
            self.time_start = time.time()
            random_number_backup = self.random_number
            while self.random_number == random_number_backup:
                self.random_number = random.randint(1,self.grid_size*self.grid_size)
        elif time.time() - self.time_start > 7.0:
            self.random_number = ""

    def draw_random_number_in_app(self,app_image):
        cv2.putText(app_image, f"{self.random_number}",(int(self.res_x/2),int(self.res_y/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


    def record(self,app_image,face_image):
        if self.last_record == "":
            self.last_record = time.time()*1000
        elif (time.time()*1000 - self.last_record) < (1/self.fps)*1000:
            return
        else:
            self.last_record = time.time()*1000

        app_image_copy = app_image.copy()
        face_image_copy = cv2.resize(face_image.copy(),[250,250])

        image_to_record = np.zeros((self.res_y, self.res_x+250, 3), np.uint8)
        image_to_record[0:0+self.res_y, 0:0+self.res_x] = app_image_copy
        image_to_record[self.res_y-250:self.res_y, self.res_x:self.res_x+250] = face_image_copy

        information_box = np.zeros((250, 250, 3), np.uint8)

        if self.demo_type == DemoType.GRID_TESTING:
            self.generate_random_number()
            self.draw_random_number_in_app(app_image)
            cv2.putText(information_box, f"Random number: {self.random_number}",(10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(information_box,  "Calibration active" if self.using_user_calibration else "Calibration inactive"  ,(10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            image_to_record[self.res_y-500:self.res_y-250, self.res_x:self.res_x+250] = information_box

        elif self.demo_type == DemoType.CALIBRATION_SCREEN:
            cv2.putText(information_box, "User calibration",(35,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            image_to_record[self.res_y-500:self.res_y-250, self.res_x:self.res_x+250] = information_box


        self.img_array.append(image_to_record.copy())

        return

    def finish(self):
        out = cv2.VideoWriter(f"{self.filename}.avi",cv2.VideoWriter_fourcc(*'DIVX'), 10, (self.img_array[0].shape[1],self.img_array[0].shape[0]))
 
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()