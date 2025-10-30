import cv2
import numpy as np
import os
from ultralytics import YOLO
import pyrealsense2 as rs

# 加载 YOLO 分割模型（保持现有模型，不做轻量化）
model = YOLO("yolo11n-seg.pt")  # 请确保模型文件在 yolo11 目录或工作目录下

# 预测参数
predict_params = {"conf": 0.4, "iou": 0.7}

# 初始化 RealSense（color + depth），并把 depth 对齐到 color
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align_to = rs.stream.color
align = rs.align(align_to)
pipeline.start(config)
# 查询 RealSense 相机内参与 depth_scale：只查询一次，避免在每帧中重复启动/停止管线
try:
    profile = pipeline.get_active_profile()
    color_profile_start = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile_start.get_intrinsics()
    intrinsic = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    # demo.py 中使用 factor_depth = camera.scale（depth / camera.scale -> 米），因此设置为 1/depth_scale
    factor_depth = float(1.0 / depth_scale)
    # 打印完整的内参矩阵和 factor_depth 以便外部脚本直接读取
    print('Queried RealSense intrinsics at startup.')
    print('intrinsic =\n%s' % intrinsic)
    print('factor_depth =', factor_depth)
except Exception as e:
    # 如果查询失败，保留变量为 None（调用端可决定回退到 meta.mat）
    intrinsic = None
    factor_depth = None
    print('Could not query RealSense intrinsics at startup, will fallback if needed. Exception:', e)


def depth_to_colormap(dimg):
    """把 uint16 depth 映射到 0-255 并上色（对有效像素归一化）。"""
    if dimg is None:
        return None
    vis = np.zeros_like(dimg, dtype=np.uint8)
    valid = dimg > 0
    if valid.any():
        dmin = int(dimg[valid].min())
        dmax = int(dimg[valid].max())
        if dmax > dmin:
            depth_norm = (dimg.astype(np.float32) - dmin) / (dmax - dmin)
            vis = (depth_norm * 255).astype(np.uint8)
        else:
            vis[valid] = 255
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

try:
    while True:
        frames = pipeline.wait_for_frames()

        # 原始 color（未对齐）和原始 depth（未对齐）
        raw_color_frame = frames.get_color_frame()
        orig_depth_frame = frames.get_depth_frame()
        if not raw_color_frame or not orig_depth_frame:
            continue
        raw_color = np.asanyarray(raw_color_frame.get_data())
        orig_depth = np.asanyarray(orig_depth_frame.get_data())

        # 对齐：把 depth 对齐到 color（color 本身不变）
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame:
            continue
        aligned_depth = np.asanyarray(aligned_depth_frame.get_data())

        # 生成对齐后的深度伪彩色用于可视化
        depth_colormap_aligned = depth_to_colormap(aligned_depth)

        # 运行 YOLO（在原始 color 上），并得到可视化图
        results = model.predict(raw_color, **predict_params)
        result = results[0]
        seg_img = result.plot()

        # 显示窗口：原始彩色、对齐后的深度伪彩色、YOLO 分割
        cv2.imshow('Color (raw)', raw_color)
        if depth_colormap_aligned is not None:
            cv2.imshow('Depth (aligned to color)', depth_colormap_aligned)
        cv2.imshow('YOLO Segmentation', seg_img)

        # 注：corr_mask 生成与显示已注释掉，按需可恢复
        # valid_depth = aligned_depth > 0
        # valid_color = np.any(aligned_color_image != 0, axis=2)
        # corr_mask = (valid_depth & valid_color).astype(np.uint8) * 255
        # cv2.imshow('Depth-Color Correspondence Mask', corr_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存对齐后的深度图（uint16 PNG）、原始 RGB、以及一张与 RGB 同分辨率的全白图
            save_dir = os.path.join('doc', 'example_data_01')
            os.makedirs(save_dir, exist_ok=True)
            color_path = os.path.join(save_dir, 'color.png')
            depth_path = os.path.join(save_dir, 'depth.png')
            white_path = os.path.join(save_dir, 'workspace_mask.png')
            # raw_color 是 BGR uint8，aligned_depth 是 uint16
            cv2.imwrite(color_path, raw_color)
            cv2.imwrite(depth_path, aligned_depth)
            # 生成并保存一张与 RGB 同分辨率的单通道全白灰度图（H,W）作为 workspace_mask
            h, w = raw_color.shape[:2]
            white = np.ones((h, w), dtype=np.uint8) * 255
            cv2.imwrite(white_path, white)

    # 已在启动时查询内参（见脚本顶部）；如需动态更新可在此处添加逻辑
           
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
