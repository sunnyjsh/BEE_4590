# -*- coding: utf-8 -*-
"""
A simplified script to test the RealSense camera installation.
This script will open a window to display the live color and depth feeds.
Press 'q' to quit.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def main():
    """
    Initializes the RealSense camera and displays the color and depth streams.
    """
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    # Check if the camera is a D400 series or L500 series
    # and set an appropriate resolution.
    if device_product_line == 'D400' or device_product_line == 'L500':
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        # For other series, you might need to find a supported resolution.
        # This is a common default.
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    print("Starting camera stream...")
    pipeline.start(config)
    print("Camera stream started. Press 'q' in the display window to quit.")

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (for visualization)
            # This converts the 16-bit depth data to an 8-bit color image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# Check for key presses
            key = cv2.waitKey(1) & 0xFF

            # If 'q' is pressed, break the loop
            if key == ord('q'):
                break
            
            # If 'a' is pressed, save the current frames
            elif key == ord('a'):
                # Create a timestamp for unique filenames
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                color_filename = f"manual_color_{timestamp}.png"
                depth_filename = f"manual_depth_{timestamp}.npy"
                
                # Save the color image (the one you see in the window)
                cv2.imwrite(color_filename, color_image)
                
                # Save the raw depth data as a numpy array.
                # This saves the actual distance data, not the colorful visualization.
                np.save(depth_filename, depth_image)
                
                print(f"Saved: {color_filename} and {depth_filename}")
    finally:
        # Stop streaming
        print("Stopping camera stream...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stream stopped and windows closed.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")


