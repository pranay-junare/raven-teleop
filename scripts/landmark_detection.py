import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from camera_realsense import get_serial_numbers, get_images, initialize_pipelines
import matplotlib.pyplot as plt
from utils import draw_axes, draw_pose_cube, map_range
import zmq
import json
import time
import math

IMG_HEIGHT = 480
IMG_WIDTH = 640
TEXT_SIZE = 0.7

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize Realsense camera
serial_numbers = get_serial_numbers()
pipelines = initialize_pipelines(serial_numbers)
serial, pipeline = next(iter(pipelines.items()))

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils

plt.ion()

# Create a single figure with 2 subplots side-by-side
fig, (ax_speed, ax_yaw) = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('black')  # Set background for the whole figure

# --- SPEED (Throttle) plot (Y-axis only) ---
ax_speed.set_facecolor('black')
ax_speed.set_xlim(-1, 1)
ax_speed.set_ylim(-100, 100)
ax_speed.axhline(0, color='green', linewidth=1)
ax_speed.axvline(0, color='green', linewidth=1)
ax_speed.spines['top'].set_color('none')
ax_speed.spines['right'].set_color('none')
ax_speed.spines['left'].set_color('green')
ax_speed.spines['bottom'].set_color('green')
ax_speed.tick_params(colors='green', labelcolor='white')
ax_speed.set_ylabel('Throttle', color='white')
ax_speed.set_title("Throttle (Speed)", color='white')
pointer_speed, = ax_speed.plot([], [], 'ro', markersize=10)

# --- YAW plot (X-axis only) ---
ax_yaw.set_facecolor('black')
ax_yaw.set_xlim(-90, 90)
ax_yaw.set_ylim(-1, 1)
ax_yaw.axhline(0, color='green', linewidth=1)
ax_yaw.axvline(0, color='green', linewidth=1)
ax_yaw.spines['top'].set_color('none')
ax_yaw.spines['right'].set_color('none')
ax_yaw.spines['left'].set_color('green')
ax_yaw.spines['bottom'].set_color('green')
ax_yaw.tick_params(colors='green', labelcolor='white')
ax_yaw.set_xlabel('Yaw', color='white')
ax_yaw.set_title("Yaw (Direction)", color='white')
pointer_yaw, = ax_yaw.plot([], [], 'ro', markersize=10)

fig.tight_layout()
fig.show()



# Initialize ZMQ for sending commands
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")
time.sleep(1)


def get_middle_point(landmarks):
    """Calculate the middle point between the wrist, index finger, and pinky finger."""
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_finger = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_finger = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    middle_point = (wrist + index_finger + pinky_finger) / 3
    return middle_point

def get_yaw_point(landmarks):
    """Calculate the middle point between the wrist, index finger, and pinky finger."""
    index_finger = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_finger = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    # f2_finger = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    # f3_finger = np.array([landmarks[13].x, landmarks[13].y, landmarks[13].z])
    # middle_point = (index_finger + pinky_finger + f2_finger +  f3_finger)/4

    middle_point = (index_finger + pinky_finger)/2
    return middle_point

def compute_yaw_angle(hand_landmarks):
    midpoint = get_yaw_point(hand_landmarks)
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
    dx = midpoint[0] - wrist[0]
    dy = midpoint[1] - wrist[1]

    # Direction vector
    angle = math.atan2(dx, -dy)  # vertical axis is (0, -1)
    return angle


def calculate_hand_pose(hand_landmarks, depth_image, color_intrinsics):
    '''
        Function to calculate hand pose (3D pose and orientation)
    '''
    # Get the 3D coordinates of wrist, index, and middle finger landmarks
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
    index_finger = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
    middle_finger = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z])

    # Test 1    
    # Convert normalized 2D coordinates to pixel coordinates
    wrist_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [wrist[0] * IMG_WIDTH, wrist[1] * IMG_HEIGHT], wrist[2])
    index_finger_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [index_finger[0] * IMG_WIDTH, index_finger[1] * IMG_HEIGHT], index_finger[2])
    middle_finger_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [middle_finger[0] * IMG_WIDTH, middle_finger[1] * IMG_HEIGHT], middle_finger[2])
    wrist_3d = np.array(wrist_2d)
    index_finger_3d = np.array(index_finger_2d)
    middle_finger_3d = np.array(middle_finger_2d)

    # Test 2
    # wrist_3d = wrist
    # index_finger_3d = index_finger
    # middle_finger_3d = middle_finger

    # Vectors AB (Index - Wrist) and AC (Middle - Wrist)
    vector_ab = index_finger_3d - wrist_3d
    vector_ac = middle_finger_3d - wrist_3d

    # Hand normal vector is the cross product of AB and AC
    vector_z = np.cross(vector_ab, vector_ac)
    vector_z = vector_z / np.linalg.norm(vector_z)  # Normalize the normal

    # Calculate "X" direction (mean of wrist, index, and middle)
    mean_point = (wrist_3d + index_finger_3d + middle_finger_3d) / 3
    vector_x = mean_point - wrist_3d
    vector_x = vector_x / np.linalg.norm(vector_x)

    # Calculate "Y" direction (cross product of normal and X)
    vector_y = np.cross(vector_z, vector_x)

    # Quaternion calculation (simplified for visualization)
    q_w = np.sqrt(1 + vector_x[0] + vector_y[1] + vector_z[2]) / 2
    q_x = (vector_y[2] - vector_x[1]) / (4 * q_w)
    q_y = (vector_x[2] - vector_z[0]) / (4 * q_w)
    q_z = (vector_z[1] - vector_y[0]) / (4 * q_w)

    # Ensure quaternion components are valid (not NaN or Inf)
    if np.any(np.isnan([q_w, q_x, q_y, q_z])) or np.any(np.isinf([q_w, q_x, q_y, q_z])):
        print("Invalid quaternion components!")
        # You can either skip processing this frame or assign a default quaternion value
        q_w, q_x, q_y, q_z = 1, 0, 0, 0  # Default identity quaternion
    else:
        # Proceed with calculations for yaw, pitch, and roll
        pitch_input = 2 * (q_w * q_y - q_z * q_x)
        pitch_input = np.clip(pitch_input, -1.0, 1.0)
        pitch = np.arcsin(pitch_input)

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll = np.arctan2(2*(q_w*q_x + q_y*q_z), 1 - 2*(q_x**2 + q_y**2))
    pitch = np.arcsin(2*(q_w*q_y - q_z*q_x))
    yaw = np.arctan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y**2 + q_z**2))

    return roll, pitch, yaw, vector_x, vector_y, vector_z, wrist_3d


# Main loop
frame_counter = 0
while True:
    time.sleep(0.1)  # Sleep for 10ms to control the loop speed
    # Get frames from the RealSense camera
    depth_image, color_image = get_images(pipeline)
    if depth_image is not None and color_image is not None:
        # Visualize the depth image
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow(f'Depth {serial}', depth_colormap)

        # Process the color frame to find hands
        rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # If hands are detected, calculate and draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Retrieve camera intrinsics for depth-to-3D conversion
                profile = pipeline.get_active_profile()
                color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Draw midpoint
                middle_point_3d  = get_middle_point(hand_landmarks.landmark)
                h, w = color_image.shape[:2]
                midpoint_x = int(middle_point_3d[0] * w)
                midpoint_y = int(middle_point_3d[1] * h)
                midpoint_x = np.clip(midpoint_x, 0, w - 1)  
                midpoint_y = np.clip(midpoint_y, 0, h - 1)  
                cv2.circle(color_image, (midpoint_x, midpoint_y), 5, (0,  0 , 255), -1) 

                # Depth calculation in mm
                depth_value = depth_image[midpoint_y, midpoint_x]
                depth_value = depth_value*0.25

                # Calculate 3D pose and orientation of the hand
                roll, pitch, yaw, x_axis, y_axis, z_axis, wrist_3d = calculate_hand_pose(hand_landmarks.landmark, depth_image, color_intrinsics)
                
                yaw = compute_yaw_angle(hand_landmarks.landmark)
                robot_speed = map_range(depth_value, 500, 800, -100, 100)  # Speed: (-100 to 100)
                robot_yaw = map_range(yaw, -np.pi, np.pi, -180, 180)  # Yaw: (-90 to 90)
                sock.send_string(json.dumps({ "robot_speed": robot_speed, "robot_yaw": robot_yaw }))                # sock.send_string(json.dumps({"action": "yaw", "speed": robot_yaw}))

                # visualization
                draw_axes(color_image, x_axis, y_axis, z_axis, (IMG_WIDTH/6, IMG_HEIGHT/1.2))
                # draw_pose_cube(color_image, yaw, pitch, roll, (IMG_WIDTH/4, IMG_HEIGHT/1.3))

                # Display the Euler angles (yaw, pitch, roll) on the frame
                cv2.putText(color_image, f"Robot Yaw: {robot_yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (0, 255, 0), 2)
                cv2.putText(color_image, f"Robot Speed: {robot_speed:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (0, 255, 0), 2)
                # cv2.putText(color_image, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (0, 255, 0), 2)
                # cv2.putText(color_image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (0, 255, 0), 2)
                # cv2.putText(color_image, f'Depth: {depth_value}mm', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (0, 255, 0), 2)
                pointer_speed.set_data([0], [robot_speed])  # Vertical plot, centered X
                pointer_yaw.set_data([robot_yaw], [0])      # Horizontal plot, centered Y

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
               
        else:
            # If no hands are detected, send stop command
            robot_speed = 0
            robot_yaw = 0
            sock.send_string(json.dumps({ "robot_speed": robot_speed, "robot_yaw": robot_yaw }))  
            pointer_speed.set_data([0], [0])    
            pointer_yaw.set_data([0], [0])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
              # sock.send_string(json.dumps({"action": "yaw", "speed": robot_yaw}))
        # Display the color image with landmarks and Euler angles
        cv2.imshow("Hand Landmark and Pose", color_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Stop the RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()
