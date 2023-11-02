from enum import Enum
from websockets import WebSocketServerProtocol
from tools.config_instance import ConfigInstance
from tools.pipeline_getter import PipelineGetter

import logging
import json
import time
import json
logging.basicConfig(level=logging.INFO)

class States(str, Enum):
    INIT = 1
    MOUSE = 2

class Server:
    def __init__(self):
        #Config
        config_instance = ConfigInstance()

        self.resolution_x = config_instance.resolution_x
        self.resolution_y = config_instance.resolution_y
        self.aspect_ratio_horizontal = config_instance.aspect_ratio_horizontal
        self.aspect_ratio_vertical = config_instance.aspect_ratio_vertical
        self.diagonal_size = config_instance.diagonal_size

        self.width_screen = config_instance.width_screen
        self.height_screen = config_instance.height_screen
        self.offset_camera = config_instance.offset_camera

        self.state = States.INIT
        self.pipeline = PipelineGetter().start()
        self.old_px = 0
        self.old_py = 0
        self.model_set = 0
        self.client = 0

    async def register(self,ws:WebSocketServerProtocol)->None:
        self.client = ws
        logging.info(f' {ws.remote_address} connects.' )
    
    async def unregister(self,ws:WebSocketServerProtocol)->None:
        self.state = States.INIT
        self.client = 0
        logging.info(f' {ws.remote_address} disconnects.' )

    async def send_to_client(self, message: str) -> None:
        if self.client != 0:
            await self.client.send(message)

    async def ws_handler(self, ws: WebSocketServerProtocol) -> None:
        await self.register(ws)

        while True:
            time.sleep(0.3)

            if self.state == States.INIT:
                await self.process_init_state(ws)
            elif self.state == States.MOUSE:
                await self.process_mouse_state(ws)
       
    async def process_init_state(self,ws):
        response_string = await ws.recv()
        state = response_string

        if state == "Init":
            object = {"result": True}
            json_ = json.dumps(object)
            await self.send_to_client(json_)
            self.state = States.MOUSE

    async def process_mouse_state(self,ws):
        px = self.pipeline.px
        py = self.pipeline.py
        eye_open = self.pipeline.eye_open
        await self.send_points(px,py,eye_open)

    async def send_points(self,px,py,eye_open) -> None:
        object = {"px":px,"py":py,"eye_open":eye_open}
        json_ = json.dumps(object)
        await self.send_to_client(json_)










