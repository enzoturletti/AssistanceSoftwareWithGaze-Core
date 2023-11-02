from tools.config_instance import ConfigInstance
from tools.depth_camera_frame_getter import DepthCameraFrameGetter
from tools.normal_camera_frame_getter import NormalCameraFrameGetter

class FrameGetter:
    def __init__(self):
        config_instance = ConfigInstance()
        use_normal_camera = config_instance.use_normal_camera

        if use_normal_camera:
            self.getter = NormalCameraFrameGetter().start()
        else:
            self.getter = DepthCameraFrameGetter().start()

        self.stopped = self.getter.stopped
        
    def get(self):
        return self.getter.frame
    
    def stop(self):
        self.getter.stop()

        

