import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from camera_realsense import get_serial_numbers, get_images, initialize_pipelines
import matplotlib.pyplot as plt
from utils import draw_axes, draw_pose_cube

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

# Initialize a plot for real-time graph of yaw, pitch, roll
fig, ax = plt.subplots()
ax.set_ylim(-180, 180)
ax.set_xlim(0, 100)  # We'll shift the graph along the x-axis to simulate real-time plotting
yaw_vals, pitch_vals, roll_vals = [], [], []
time_vals = []


def get_middle_point(landmarks):
    """Calculate the middle point between the wrist, index finger, and pinky finger."""
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_finger = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_finger = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    middle_point = (wrist + index_finger + pinky_finger) / 3
    return middle_point


def project_point_to_pixel(point_3d, intrinsics):
    """Project 3D point into 2D image plane."""
    x = (point_3d[0] / point_3d[2]) * intrinsics.fx + intrinsics.ppx
    y = (point_3d[1] / point_3d[2]) * intrinsics.fy + intrinsics.ppy
    return int(x), int(y)


def draw_hand_square_on_black(wrist_3d, x_axis, y_axis, z_axis, intrinsics, image_size=(720, 1280), scale=0.1, color=(0, 255, 0), thickness=2):
    """
    Draws a square on a black background based on the hand's local axes.
    
    Args:
        wrist_3d: The 3D position of the wrist (numpy array of shape (3,)).
        x_axis, y_axis, z_axis: The local 3D unit vectors (numpy arrays of shape (3,)).
        intrinsics: RealSense camera intrinsics for projection.
        image_size: Size of the black background (height, width).
        scale: Size of the square.
        color: Line color for drawing.
        thickness: Thickness of the lines.
        
    Returns:
        The generated black image with the drawn square.
    """
    # Create a black image
    black_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Calculate the 4 corner points in 3D space
    corner1 = wrist_3d + scale * ( x_axis + y_axis)
    corner2 = wrist_3d + scale * ( x_axis - y_axis)
    corner3 = wrist_3d + scale * (-x_axis - y_axis)
    corner4 = wrist_3d + scale * (-x_axis + y_axis)

    # Project the 3D points to 2D
    corner1_2d = project_point_to_pixel(corner1, intrinsics)
    corner2_2d = project_point_to_pixel(corner2, intrinsics)
    corner3_2d = project_point_to_pixel(corner3, intrinsics)
    corner4_2d = project_point_to_pixel(corner4, intrinsics)

    # Draw the square by connecting the corners
    cv2.line(black_image, corner1_2d, corner2_2d, color, thickness)
    cv2.line(black_image, corner2_2d, corner3_2d, color, thickness)
    cv2.line(black_image, corner3_2d, corner4_2d, color, thickness)
    cv2.line(black_image, corner4_2d, corner1_2d, color, thickness)

    return black_image


# Function to calculate hand pose (3D pose and orientation)
def calculate_hand_pose(hand_landmarks, depth_image, color_intrinsics):
    # Get the 3D coordinates of wrist, index, and middle finger landmarks
    wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z])
    index_finger = np.array([hand_landmarks[5].x, hand_landmarks[5].y, hand_landmarks[5].z])
    middle_finger = np.array([hand_landmarks[9].x, hand_landmarks[9].y, hand_landmarks[9].z])

    # Convert normalized 2D coordinates to pixel coordinates (assuming 640x480 resolution)
    wrist_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [wrist[0] * 640, wrist[1] * 480], wrist[2])
    index_finger_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [index_finger[0] * 640, index_finger[1] * 480], index_finger[2])
    middle_finger_2d = rs.rs2_deproject_pixel_to_point(color_intrinsics, [middle_finger[0] * 640, middle_finger[1] * 480], middle_finger[2])

    wrist_3d = np.array(wrist_2d)
    index_finger_3d = np.array(index_finger_2d)
    middle_finger_3d = np.array(middle_finger_2d)

    # Vectors AB (Index - Wrist) and AC (Middle - Wrist)
    vector_ab = index_finger_3d - wrist_3d
    vector_ac = middle_finger_3d - wrist_3d

    # Hand normal vector is the cross product of AB and AC
    hand_normal = np.cross(vector_ab, vector_ac)
    hand_normal = hand_normal / np.linalg.norm(hand_normal)  # Normalize the normal

    # Calculate "X" direction (mean of wrist, index, and middle)
    mean_point = (wrist_3d + index_finger_3d + middle_finger_3d) / 3
    vector_x = mean_point - wrist_3d
    vector_x = vector_x / np.linalg.norm(vector_x)

    # Calculate "Y" direction (cross product of normal and X)
    vector_y = np.cross(hand_normal, vector_x)

    # Quaternion calculation (simplified for visualization)
    q_w = np.sqrt(1 + vector_x[0] + vector_y[1] + hand_normal[2]) / 2
    q_x = (vector_y[2] - vector_x[1]) / (4 * q_w)
    q_y = (vector_x[2] - hand_normal[0]) / (4 * q_w)
    q_z = (hand_normal[1] - vector_y[0]) / (4 * q_w)

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    roll = np.arctan2(2*(q_w*q_x + q_y*q_z), 1 - 2*(q_x**2 + q_y**2))
    pitch = np.arcsin(2*(q_w*q_y - q_z*q_x))
    yaw = np.arctan2(2*(q_w*q_z + q_x*q_y), 1 - 2*(q_y**2 + q_z**2))

    return roll, pitch, yaw, vector_x, vector_y, hand_normal, wrist_3d


# Main loop
frame_counter = 0
while True:
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
                cv2.circle(color_image, (midpoint_x, midpoint_y), 5, (0,  0 , 255), -1) 

                # Depth 
                depth_value = depth_image[midpoint_y, midpoint_x]
                cv2.putText(color_image, f'Depth: {depth_value}mm', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Calculate 3D pose and orientation of the hand
                roll, pitch, yaw, x_axis, y_axis, z_axis, wrist_3d = calculate_hand_pose(hand_landmarks.landmark, depth_image, color_intrinsics)
                
                # visualization
                black_image = np.zeros((480, 640, 3), dtype=np.uint8)
                draw_axes(black_image, (640/2, 480/2,), x_axis, y_axis, z_axis)
                cv2.imshow("Square on Black", black_image)

                # Display the Euler angles (yaw, pitch, roll) on the frame
                cv2.putText(color_image, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, f"Pitch: {pitch:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, f"Roll: {roll:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                #TODO: fix the bug
                # draw_pose_cube(color_image, yaw, pitch, roll)


                # Append values to the real-time graph list
                yaw_vals.append(yaw)
                pitch_vals.append(pitch)
                roll_vals.append(roll)
                time_vals.append(frame_counter)

                # Update the plot
                if len(yaw_vals) > 100:  # Limit the number of data points to 100 for a smoother graph
                    yaw_vals.pop(0)
                    pitch_vals.pop(0)
                    roll_vals.pop(0)
                    time_vals.pop(0)

                # Plotting in real-time
                ax.clear()
                ax.plot(time_vals, yaw_vals, label="Yaw", color='r')
                ax.plot(time_vals, pitch_vals, label="Pitch", color='g')
                ax.plot(time_vals, roll_vals, label="Roll", color='b')
                ax.legend(loc='upper left')
                plt.pause(0.01)  # Update the graph in real-time

        # Display the color image with landmarks and Euler angles
        cv2.imshow("Hand Landmark and Pose", color_image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_counter += 1

# Stop the RealSense pipeline
pipeline.stop()
cv2.destroyAllWindows()
