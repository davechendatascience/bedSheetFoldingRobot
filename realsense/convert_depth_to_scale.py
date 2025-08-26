import os
import numpy as np
import pyrealsense2 as rs

# === CONFIGURATION ===
input_dir = "realsense_camera/"      # folder with *.npy files (raw depth)
output_dir = "depth_scaled"  # will contain meter-scale *.npy files

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# === Obtain pyrealsense2 depth_scale ===

# This code assumes you have a RealSense device connected.
# If you know your scale (e.g., 0.001), set manually; otherwise, read from device:
pipeline = rs.pipeline()
config = rs.config()
pipeline_profile = pipeline.start(config)
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
pipeline.stop()

print("Using RealSense depth_scale:", depth_scale)

# === Process .npy files ===

n_files = 0
for fname in os.listdir(input_dir):
    if fname.endswith('.npy'):
        fin_path = os.path.join(input_dir, fname)
        fout_path = os.path.join(output_dir, fname)
        depth_raw = np.load(fin_path)
        depth_m = depth_raw.astype(np.float32) * depth_scale  # in meters
        np.save(fout_path, depth_m)
        n_files += 1
        print(f"Converted: {fname} -> {fout_path}")

print(f"Done! {n_files} files processed. All depths written in meters.")
