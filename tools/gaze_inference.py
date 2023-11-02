import abc

class GazeInference(metaclass=abc.ABCMeta):
    def __init__(self):
        self.load_model()
        self.load_transform()


    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def load_transform(self):
        pass

    @abc.abstractmethod
    def inference(self, normalizated_img):
        pass

