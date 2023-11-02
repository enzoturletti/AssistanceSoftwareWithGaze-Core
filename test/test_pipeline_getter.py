import cv2
import numpy as np
from tools.pipeline_getter import PipelineGetter
from tools.grid_screen import GridDisplay
from tools.config_instance import ConfigInstance
from tools.video_demostration_builder import VideoDemostrationBuilder
from tools.video_demostration_builder import DemoType

config_instance = ConfigInstance()
pipeline_getter = PipelineGetter().start()
grid_display = GridDisplay()

video_demo_activated = config_instance.video_demo_activated
if config_instance.video_demo_activated: video_demo_build = VideoDemostrationBuilder("test_gaze_model_grid_performance",DemoType.GRID_TESTING)


while True:
    if (cv2.waitKey(1) & 0xFF == 27) or pipeline_getter.stopped:
        pipeline_getter.stop()
        if config_instance.video_demo_activated: video_demo_build.finish()
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

    if config_instance.video_demo_activated: video_demo_build.record(grid,pipeline_getter.frame_getter.get())

    cv2.namedWindow("Grid", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Grid",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Grid", grid)



