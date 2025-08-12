import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

# save path
save_image_dir = "realsense_camera/"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)  # Match resolutions

pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

def get_aligned_frames():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Normalize for display
    if np.any(depth_image):
        dmin = np.min(depth_image[np.nonzero(depth_image)])
        dmax = np.max(depth_image)
        depth_disp = ((depth_image - dmin) / (dmax - dmin) * 255).astype(np.uint8) if dmax > dmin and dmin > 0 else np.zeros_like(depth_image, dtype=np.uint8)
    else:
        depth_disp = np.zeros_like(depth_image, dtype=np.uint8)
    return color_image, depth_image, depth_disp

class RGBDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB & Depth Capture")
        self.label = tk.Label(root)
        self.label.pack()
        self.button = tk.Button(root, text="Photo Shoot", command=self.save_images)
        self.button.pack()
        self.color = None
        self.depth = None
        self.update()

    def update(self):
        c, d, d_disp = get_aligned_frames()
        if c is not None and d_disp is not None:
            self.color = c
            self.depth = d
            d_rgb = cv2.cvtColor(d_disp, cv2.COLOR_GRAY2RGB)
            stack = np.hstack((c, d_rgb))
            img = Image.fromarray(stack)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
        self.root.after(30, self.update)

    def save_images(self):
        if self.color is not None and self.depth is not None:
            ts = int(time.time())
            cv2.imwrite(f"{save_image_dir}color_{ts}.png", self.color)
            cv2.imwrite(f"{save_image_dir}depth_{ts}.png", self.depth)
            print(f"Saved: color_{ts}.png and depth_{ts}.png")

root = tk.Tk()
app = RGBDepthApp(root)
try:
    root.mainloop()
finally:
    pipeline.stop()
