import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import timm

import numpy as np
from PIL import Image

from tools.config_instance import ConfigInstance
from tools.config_instance import ConfigInstance
from tools.gaze_inference import GazeInference
from archs.arch import myModel

class GazeModelInference(GazeInference):
    def __init__(self):
        super().__init__()

    def load_model(self):
        config_instance = ConfigInstance()
        model_path = config_instance.gaze_model_path
        model = myModel(eth_xgaze_pretrain=False)
        checkpoint = torch.load(model_path,map_location='cpu')    
        model.load_state_dict(checkpoint)
        model = model.to("cpu")
        model.eval()
        model.train = False
        self.model = model

    def load_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def inference(self, normalizated_image):
        with torch.no_grad():
            normalizated_image  = Image.fromarray(normalizated_image)
            normalizated_image  = self.transform(normalizated_image)
            normalizated_image  = Variable(normalizated_image)
            normalizated_image  = normalizated_image.unsqueeze(0) 
            
            yaw_predicted, pitch_predicted = self.model(normalizated_image)
            yaw_predicted = yaw_predicted.cpu().numpy()
            pitch_predicted = pitch_predicted.cpu().numpy()

            return yaw_predicted, pitch_predicted
        