import cv2
import numpy as np
from tools.config_instance import ConfigInstance
class GridDisplay:
    def __init__(self):
        config_instance = ConfigInstance()
        
        self.res_x = config_instance.resolution_x
        self.res_y = config_instance.resolution_y
        self.grid_size = config_instance.grid_size
        self.grid = np.zeros((self.res_y, self.res_x, 3), np.uint8)
        self.draw_grid()

    def draw_grid(self):
        width, height = self.res_x, self.res_y
        grid_width = width // self.grid_size
        grid_height = height // self.grid_size

        for i in range(self.grid_size + 1):
            start_point_vertical = (i * grid_width, 0)
            end_point_vertical = (i * grid_width, height)
            start_point_horizontal = (0, i * grid_height)
            end_point_horizontal = (width, i * grid_height)

            cv2.line(self.grid, start_point_vertical, end_point_vertical, (255, 255, 255), 1)
            cv2.line(self.grid, start_point_horizontal, end_point_horizontal, (255, 255, 255), 1)

        # Add numbers to rectangles.
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cv2.putText(self.grid, f"{i*self.grid_size+j+1}",(j * grid_width + 10,i * grid_height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(self.grid, (width - 1, 0), (width - 1, height), (255, 255, 255), 1)
        cv2.line(self.grid, (0, height - 1), (width, height - 1), (255, 255, 255), 1)

    def color_grid_element(self, px, py, color = (0,255,0)):
        width, height = self.res_x, self.res_y
        grid_width = width // self.grid_size
        grid_height = height // self.grid_size

        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))

        grid_x = px // grid_width
        grid_y = py // grid_height

        top_left = (grid_x * grid_width, grid_y * grid_height)
        bottom_right = ((grid_x + 1) * grid_width - 1, (grid_y + 1) * grid_height - 1)

        self.temp_img = self.grid.copy()
        cv2.rectangle(self.temp_img, top_left, bottom_right, color, -1)
        cv2.circle(self.temp_img, (int(px),int(py)) , radius=5, color=(255, 255, 255), thickness=-1)

    def get_grid(self):
        return self.temp_img

