import zmq
import json
import time

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")
time.sleep(1)

while True:
    sock.send_string(json.dumps({"action": "forward", "speed": 0.1}))
    time.sleep(0.1)

# Move forward
# sock.send_string(json.dumps({"action": "forward", "speed": 0.1}))

# Rotate    
# sock.send_string(json.dumps({"action": "yaw", "speed": 0.5}))

