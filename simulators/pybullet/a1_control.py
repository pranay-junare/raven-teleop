import numpy as np
import time
import threading
import zmq
import json
from src.State import BehaviorState, State
from src.Command import Command
from src.Utilities import deadband, clipped_first_order_filter
from vjoy import VJoy

MESSAGE_RATE = 20

class JoystickInterface:
    def __init__(self, config, udp_port=8830, udp_publisher_port=8840):
        self.config = config
        self.previous_gait_toggle = 0
        self.previous_state = BehaviorState.REST
        self.previous_hop_toggle = 0
        self.previous_activate_toggle = 0

        self.message_rate = MESSAGE_RATE
        self.joystick = VJoy(callback=None)

        reader = threading.Thread(target=self.joystick.start_open_loop)
        reader.daemon = True
        reader.start()

        # Initialize ZMQ subscriber
        self.zmq_ctx = zmq.Context()
        self.zmq_sub = self.zmq_ctx.socket(zmq.SUB)
        self.zmq_sub.connect("tcp://localhost:5555")
        self.zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_sub.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        self.latest_zmq_msg = {}

    def scale_input(val, in_deadzone=20, in_max=100, out_min=0.5, out_max=2.8):
    # Deadzone
        if -in_deadzone <= val <= in_deadzone:
            return 0.0

        # Clamp input to max range
        val = max(min(val, in_max), -in_max)

        # Scale linearly outside the deadzone
        sign = 1 if val > 0 else -1
        val = abs(val)

        scaled = ((val - in_deadzone) / (in_max - in_deadzone)) * (out_max - out_min) + out_min
        return scaled * sign

    def poll_zmq(self):
        try:
            msg = self.zmq_sub.recv_string()
            self.latest_zmq_msg = json.loads(msg)
        except zmq.Again:
            pass  # no message received

    def get_command(self, state, do_print=False):
        try:
            msg = self.get_joystick()
            self.poll_zmq()
            # ZMQ override logic
            if "forward" in self.latest_zmq_msg:
                #print(self.latest_zmq_msg["forward"] / self.config.max_x_velocity)
                msg["ly"] = self.latest_zmq_msg["forward"]
            
            if "Yaw" in self.latest_zmq_msg:
                if(self.latest_zmq_msg["Yaw"] > 12):
                    msg["rx"] = 1
                elif(self.latest_zmq_msg["Yaw"] < -12):
                     msg["rx"] = -1
                else:
                    msg["rx"] = 0
            print("Calclllllllll",msg["rx"])
            print("actuaaalalalla", self.latest_zmq_msg["Yaw"])
            command = Command()

            # Discrete state transitions
            gait_toggle = msg["R1"]
            command.trot_event = gait_toggle == 1 and self.previous_gait_toggle == 0

            hop_toggle = msg["x"]
            command.hop_event = hop_toggle == 1 and self.previous_hop_toggle == 0

            activate_toggle = msg["L1"]
            command.activate_event = activate_toggle == 1 and self.previous_activate_toggle == 0

            self.previous_gait_toggle = gait_toggle
            self.previous_hop_toggle = hop_toggle
            self.previous_activate_toggle = activate_toggle

            # Continuous commands

            x_vel = JoystickInterface.scale_input(msg["ly"]) * self.config.max_x_velocity

            y_vel = msg["lx"] * -self.config.max_y_velocity
            command.horizontal_velocity = np.array([x_vel, y_vel])
            command.yaw_rate = msg["rx"] * -self.config.max_yaw_rate


            message_rate = msg["message_rate"]
            message_dt = 1.0 / message_rate

            pitch = msg["ry"] * self.config.max_pitch
            deadbanded_pitch = deadband(pitch, self.config.pitch_deadband)
            pitch_rate = clipped_first_order_filter(
                state.pitch,
                deadbanded_pitch,
                self.config.max_pitch_rate,
                self.config.pitch_time_constant,
            )
            command.pitch = state.pitch + message_dt * pitch_rate

            height_movement = msg["dpady"]
            command.height = state.height - message_dt * self.config.z_speed * height_movement

            roll_movement = -msg["dpadx"]
            command.roll = state.roll + message_dt * self.config.roll_speed * roll_movement

            return command

        except Exception as e:
            if do_print:
                print("Error in get_command:", e)
            return Command()

    def get_joystick(self):
        buttons = self.joystick.buttons
        axes = self.joystick.axes

        left_y = -axes[1] / 32767.0
        right_y = -axes[4] / 32767.0
        right_x = axes[3] / 32767.0
        left_x = axes[0] / 32767.0

        L2 = axes[2] / 32767.0
        R2 = axes[5] / 32767.0

        R1 = buttons[5]
        L1 = buttons[4]

        square = buttons[2]
        x = buttons[0]
        circle = buttons[1]
        triangle = buttons[3]

        dpadx = axes[6] / 32767.0
        dpady = -axes[7] / 32767.0

        msg = {
            "ly": left_y,
            "lx": left_x,
            "rx": right_x,
            "ry": right_y,
            "L2": L2,
            "R2": R2,
            "R1": R1,
            "L1": L1,
            "dpady": dpady,
            "dpadx": dpadx,
            "x": x,
            "square": square,
            "circle": circle,
            "triangle": triangle,
            "message_rate": self.message_rate,
        }
        return msg

    def set_color(self, color):
        joystick_msg = {"ps4_color": color}
        # self.udp_publisher.send(joystick_msg)  # Uncomment if needed

