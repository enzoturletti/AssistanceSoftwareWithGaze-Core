from tools.server import Server
import websockets
import asyncio


server = Server()
start_server = websockets.serve(server.ws_handler,"localhost",2223)
loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
loop.run_forever()