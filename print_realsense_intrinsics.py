#!/usr/bin/env python3
"""Print RealSense intrinsics (color) and factor_depth for color-aligned depth.

Usage: python print_realsense_intrinsics.py

Notes:
- Uses color stream intrinsics (fx, fy, cx, cy) because depth is aligned to color.
- factor_depth = 1.0 / depth_scale, suitable for CameraInfo(scale=factor_depth).
"""

import sys
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    print("pyrealsense2 not available:", e)
    sys.exit(1)


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    # Request common, safe defaults; alignment will warp depth to color resolution
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
    except Exception:
        # Fallback to device defaults if the requested profile isn't supported
        profile = pipeline.start()

    align = rs.align(rs.stream.color)

    try:
        # Warm-up a couple frames
        for _ in range(5):
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print("Failed to acquire color/depth frames")
            return 1

        # Color intrinsics (used for color-aligned depth)
        color_vsp = color_frame.get_profile().as_video_stream_profile()
        intr = color_vsp.get_intrinsics()
        fx, fy = float(intr.fx), float(intr.fy)
        cx, cy = float(intr.ppx), float(intr.ppy)
        width, height = int(intr.width), int(intr.height)

        # Depth scale and factor_depth
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())  # meters per unit
        factor_depth = 1.0 / depth_scale                     # units per meter

        # Print results
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        np.set_printoptions(precision=6, suppress=True)

        print("=== RealSense (color) intrinsics for color-aligned depth ===")
        print(f"width:  {width}")
        print(f"height: {height}")
        print(f"fx: {fx:.6f}, fy: {fy:.6f}")
        print(f"cx: {cx:.6f}, cy: {cy:.6f}")
        print("intrinsic matrix K =")
        print(K)
        print()
        print(f"depth_scale (meters per unit): {depth_scale:.9f}")
        print(f"factor_depth (units per meter): {factor_depth:.6f}")
        print()
        print("Paste into code:")
        print(f"intrinsic = np.array([[{fx:.6f}, 0.0, {cx:.6f}], [0.0, {fy:.6f}, {cy:.6f}], [0.0, 0.0, 1.0]], dtype=np.float32)")
        print(f"camera_info = CameraInfo({float(width)}, {float(height)}, {fx:.6f}, {fy:.6f}, {cx:.6f}, {cy:.6f}, {factor_depth:.6f})")

        return 0
    finally:
        pipeline.stop()


if __name__ == "__main__":
    sys.exit(main())
