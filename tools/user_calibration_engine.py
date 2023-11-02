from cgi import print_arguments
import json
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from tools.config_instance import ConfigInstance

class UserCalibrationEngine:
    def __init__(self):
        
        config_instance = ConfigInstance()
        path = config_instance.calibration_gaze_model_output
        file = open(path, 'r')
        lines = file.readlines()

        real_coords = []
        measured_coords = []

        for line in lines:
            data_dictionary = json.loads(line.replace("'",'"'))
            coordinates_in_screen = np.array([data_dictionary["px"], data_dictionary["py"]])
            real_coords.append(coordinates_in_screen)
            coordinates_in_screen = np.array([data_dictionary["px_estimated"], data_dictionary["py_estimated"]])
            measured_coords.append(coordinates_in_screen)

        self.real_coords = np.array(real_coords)
        self.measured_coords = np.array(measured_coords)


        
        # Create a polynomial regression model
        degree = 3 # adjust the degree to suit your needs
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(self.measured_coords, self.real_coords)

        self.model = model

    def correct(self,px,py):
        measure_coord = np.array([px,py])
        result = self.model.predict(measure_coord.reshape(1,-1))
        
        corrected_px = result[0][0]
        corrected_py = result[0][1]

        return corrected_px, corrected_py


        

        
