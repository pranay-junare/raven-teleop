import pyrealsense2 as rs
import cv2
import numpy as np

CAMERA_INVERTED = False
CAMERA_FLIP = True

def get_serial_numbers():
    context = rs.context()
    devices = context.query_devices()
    serial_numbers = [device.get_info(rs.camera_info.serial_number) for device in devices]
    return serial_numbers

def initialize_pipelines(serial_numbers):
    pipelines = {}
    for serial in serial_numbers:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipelines[serial] = pipeline
    return pipelines

def get_images(pipeline):
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame or not color_frame:
        return None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    if CAMERA_INVERTED == True:
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
    if CAMERA_FLIP == True:
        depth_image = cv2.flip(depth_image, 1)
        color_image = cv2.flip(color_image, 1)
    return depth_image, color_image


if __name__ == '__main__':
    serial_numbers = get_serial_numbers()
    print("Total number of cameras connected: ", len(serial_numbers))
    print("Serial numbers of connected cameras: ", serial_numbers)

    pipelines = initialize_pipelines(serial_numbers)

    try:
        while True:
            for serial, pipeline in pipelines.items():
                depth_image, color_image = get_images(pipeline)
                if depth_image is not None and color_image is not None:
                    cv2.imshow(f'Depth {serial}', depth_image)
                    cv2.imshow(f'Color {serial}', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        cv2.destroyAllWindows()
