from math import cos, sin
import numpy as np
import cv2

def draw_axes(image, x_axis, y_axis, z_axis, origin, length=50):
    """
    Draws 3D axes (X, Y, Z) on the image with respect to the origin.
    """
    origin = tuple(map(int, origin))
    x_end = tuple(map(int, (origin[0] + length * x_axis[0], origin[1] - length * x_axis[1])))
    y_end = tuple(map(int, (origin[0] + length * y_axis[0], origin[1] - length * y_axis[1])))
    z_end = tuple(map(int, (origin[0] + length * z_axis[0], origin[1] - length * z_axis[1])))

    cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 2, tipLength=0.2)  # X - Red
    cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 2, tipLength=0.2)  # Y - Green
    cv2.arrowedLine(image, origin, z_end, (255, 0, 0), 2, tipLength=0.2)  # Z - Blue
    return image


def draw_pose_cube(img, y, p, r, origin = (0.5, 0.5), size=50., in_degrees=False):
    """
    Draws a 3D pose cube on the image. The cube's origin is passed and is aligned with the 3D axes.
    """
    if in_degrees:
        p = p * np.pi / 180
        y = -(y * np.pi / 180)
        r = r * np.pi / 180

    # Adjust translation (center_x, center_y) relative to origin
    center_x = origin[0] - 0.50 * size
    center_y = origin[1] - 0.50 * size

    # Calculate cube corners based on yaw, pitch, roll
    x1 = size * (cos(y) * cos(r)) + center_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + center_y
    x2 = size * (-cos(y) * sin(r)) + center_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + center_y
    x3 = size * (sin(y)) + center_x
    y3 = size * (-cos(y) * sin(p)) + center_y

    # Draw cube base in red
    cv2.line(img, (int(center_x), int(center_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(center_x), int(center_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - center_x), int(y2 + y1 - center_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - center_x), int(y1 + y2 - center_y)), (0, 0, 255), 3)

    # Draw pillars in blue
    cv2.line(img, (int(center_x), int(center_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - center_x), int(y1 + y3 - center_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - center_x), int(y2 + y3 - center_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - center_x), int(y2 + y1 - center_y)),
             (int(x3 + x1 + x2 - 2 * center_x), int(y3 + y2 + y1 - 2 * center_y)), (255, 0, 0), 2)

    # Draw top in green
    cv2.line(img, (int(x3 + x1 - center_x), int(y3 + y1 - center_y)),
             (int(x3 + x1 + x2 - 2 * center_x), int(y3 + y2 + y1 - 2 * center_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - center_x), int(y2 + y3 - center_y)),
             (int(x3 + x1 + x2 - 2 * center_x), int(y3 + y2 + y1 - 2 * center_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - center_x), int(y3 + y1 - center_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - center_x), int(y3 + y2 - center_y)), (0, 255, 0), 2)

    return img


def map_range(value, in_min, in_max, out_min, out_max):
    """
    Maps a value from one range to another.

    Args:
        value (float): Input value to map.
        in_min (float): Minimum of the input range.
        in_max (float): Maximum of the input range.
        out_min (float): Minimum of the output range.
        out_max (float): Maximum of the output range.

    Returns:
        float: Mapped value in the output range.
    """
    # Clamp input value to input range
    value = max(min(value, in_max), in_min)
    
    # Perform linear mapping
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min