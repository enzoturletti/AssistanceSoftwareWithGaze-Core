import cv2
import numpy as np
from tools.pipeline_getter_with_profiler import PipelineGetterWithProfiler
from tools.grid_screen import GridDisplay

pipeline_getter = PipelineGetterWithProfiler().start()
grid_display = GridDisplay()


while True:
    if (cv2.waitKey(1) & 0xFF == 27) or pipeline_getter.stopped:
        pipeline_getter.stop()
        break
    if not isinstance(pipeline_getter.frame_getter.get(),np.ndarray):
        print("Skip")
        continue

    px = pipeline_getter.px
    py = pipeline_getter.py
    eye_open = pipeline_getter.eye_open

    if eye_open:
        grid_display.color_grid_element(int(px),int(py))
    else:    
        grid_display.color_grid_element(int(px),int(py),color = (255,0,0))

    grid = grid_display.get_grid()

    cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Grid",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Grid", grid)



