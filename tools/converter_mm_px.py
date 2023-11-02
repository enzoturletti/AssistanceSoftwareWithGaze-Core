from tools.config_instance import ConfigInstance
class Converter:
    def __init__(self):
        config_instance = ConfigInstance()

        res_x = config_instance.resolution_x
        res_y = config_instance.resolution_y
        self.width_screen = config_instance.width_screen
        self.height_screen = config_instance.height_screen
        self.offset_camera = config_instance.offset_camera
        self.pixel_width_by_mm = res_x / self.width_screen   
        self.pixel_height_by_mm = res_y / self.height_screen 

    def get_pixels(self,x_cam_coordinates_system_mm,y_cam_coordinates_system_mm):
        x_in_screen_coordinate_system_mm = self.width_screen/2 - x_cam_coordinates_system_mm
        y_in_screen_coordinate_system_mm = y_cam_coordinates_system_mm + self.offset_camera
        px = int(x_in_screen_coordinate_system_mm*self.pixel_width_by_mm )
        py = int(y_in_screen_coordinate_system_mm*self.pixel_height_by_mm)

        return px, py

    def get_gt_in_camera_coordinates_system(self,px,py):
        x_in_screen_coordinate_system_mm = px/self.pixel_width_by_mm
        y_in_screen_coordinate_system_mm = py/self.pixel_height_by_mm
        x_cam_coordinates_system_mm = self.width_screen/2 - x_in_screen_coordinate_system_mm
        y_cam_coordinates_system_mm = y_in_screen_coordinate_system_mm-self.offset_camera

        return x_cam_coordinates_system_mm, y_cam_coordinates_system_mm


