import cv2
import json
import os
import numpy as np
from tools.config_instance import ConfigInstance
class BlinkDetector():
    def __init__(self):
        config_instance = ConfigInstance()

        try:
            click_calibration_path = os.path.join(config_instance.click_calibration_path,"click_calibrations_results.txt")
            click_calibration_file = open(click_calibration_path, 'r')
            line = click_calibration_file.readline()
            data_dictionary = json.loads(line.replace("'",'"'))
            self.metric = data_dictionary["metric"]

            self.reference_open_eye_1 = config_instance.reference_open_eye_1
            self.reference_open_eye_2 = config_instance.reference_open_eye_2
            self.reference_open_eye_3 = config_instance.reference_open_eye_3
            self.reference_open_eye_4 = config_instance.reference_open_eye_4
            self.reference_open_eye_5 = config_instance.reference_open_eye_5
            self.reference_close_eye =  config_instance.reference_close_eye
            self.error = False

            # Cargar imagen de referencia de ojo cerrado y convertir a escala de grises
            self.ref_cerrado = cv2.imread(self.reference_close_eye, cv2.IMREAD_GRAYSCALE)

            # Cargar imagen de referencia de ojo abierto y convertir a escala de grises
            self.ref_abierto_1 = cv2.imread(self.reference_open_eye_1, cv2.IMREAD_GRAYSCALE)
            self.ref_abierto_2 = cv2.imread(self.reference_open_eye_2, cv2.IMREAD_GRAYSCALE)
            self.ref_abierto_3 = cv2.imread(self.reference_open_eye_3, cv2.IMREAD_GRAYSCALE)
            self.ref_abierto_4 = cv2.imread(self.reference_open_eye_4, cv2.IMREAD_GRAYSCALE)
            self.ref_abierto_5 = cv2.imread(self.reference_open_eye_5, cv2.IMREAD_GRAYSCALE)

        except:
            print("Error: realice calibration.")
            self.error = True

    def det_right_eye(self,face_image,landmarks_2d):
        re_x_init = int(landmarks_2d[33][0])-5
        re_x_end =  int(landmarks_2d[133][0])+5
        re_y_init = int(landmarks_2d[159][1])-5
        re_y_end =  int(landmarks_2d[145][1])+5

        eye_image = face_image[re_y_init:re_y_end, re_x_init:re_x_end]
        eye_image = cv2.resize(eye_image,[100,100])

        return eye_image
    
    def det_left_eye(self, face_image, landmarks_2d):
        le_x_init = int(landmarks_2d[362][0])-5
        le_x_end =  int(landmarks_2d[263][0])+5
        le_y_init = int(landmarks_2d[386][1])-5
        le_y_end =  int(landmarks_2d[374][1])+5

        eye_image = face_image[le_y_init:le_y_end, le_x_init:le_x_end]
        eye_image = cv2.resize(eye_image,[100,100])

        return eye_image

    def detect(self,face_image,landmarks_2d):
        if self.error == True:
            print("Realizate click calibration")
            return
        
        eye = self.det_right_eye(face_image,landmarks_2d)
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

        # Calcular la correlación de Pearson entre la imagen de entrada y la imagen de referencia del ojo cerrado y del ojo abierto
        corr_abierto_1 = cv2.matchTemplate(gray_eye, self.ref_abierto_1, cv2.TM_CCOEFF_NORMED)
        corr_abierto_2 = cv2.matchTemplate(gray_eye, self.ref_abierto_2, cv2.TM_CCOEFF_NORMED)
        corr_abierto_3 = cv2.matchTemplate(gray_eye, self.ref_abierto_3, cv2.TM_CCOEFF_NORMED)
        corr_abierto_4 = cv2.matchTemplate(gray_eye, self.ref_abierto_4, cv2.TM_CCOEFF_NORMED)
        corr_abierto_5 = cv2.matchTemplate(gray_eye, self.ref_abierto_5, cv2.TM_CCOEFF_NORMED)
        corr_cerrado   = cv2.matchTemplate(gray_eye, self.ref_cerrado, cv2.TM_CCOEFF_NORMED)

        max_corr_cerrado = np.max(corr_cerrado)*self.metric
        max_corr_abierto = np.max([np.max(corr_abierto_1),np.max(corr_abierto_2),np.max(corr_abierto_3),np.max(corr_abierto_4),np.max(corr_abierto_5)])

        if(max_corr_abierto > max_corr_cerrado):
            return True
        else:
            return False

