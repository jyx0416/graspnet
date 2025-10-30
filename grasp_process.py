"""实时 GraspNet 演示（基于 Intel RealSense D435i）。

该脚本启动 RealSense 摄像头（颜色与深度对齐），由对齐后的深度生成点云，
对采样点运行 GraspNet 模型，并在界面中可视化预测到的抓取（3D gripper / 2D 调试窗口）。

设计要点：
- 重用仓库内已有函数与类（模型、数据工具、graspnetAPI）。
- 保持与 `demo.py` 的处理流程一致，但改为对实时帧处理。
- 可视化上同时保留 2D 调试窗口（RGB/深度伪彩色）和 Open3D 三维抓手显示。

用法示例：
    python realsense_grasp_demo.py --checkpoint_path path/to/checkpoint.pth

依赖：pyrealsense2、opencv-python、torch 及仓库内其他依赖。
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import open3d as o3d
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnetAPI'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from utils.collision_detector import ModelFreeCollisionDetector
# 轻量级目标检测/分割：使用仓库中 yolo11 的 ultralytics 接口
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False

import pyrealsense2 as rs


def safe_transform(geom, T):
    try:
        geom.transform(T)
    except Exception:
        pass


def update_pcd(pcd, cloud_masked, color_masked, T):
    if cloud_masked is not None and len(cloud_masked) > 0:
        pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        safe_transform(pcd, T)
    else:
        pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float32))
        pcd.colors = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float32))
        safe_transform(pcd, T)


def replace_grippers(vis, prev_gripper_geoms, grippers, T):
    for g in prev_gripper_geoms:
        try:
            vis.remove_geometry(g)
        except Exception:
            pass
    new_geoms = []
    for g in grippers:
        safe_transform(g, T)
        vis.add_geometry(g)
        new_geoms.append(g)
    return new_geoms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000)
    parser.add_argument('--num_view', type=int, default=300)
    parser.add_argument('--collision_thresh', type=float, default=0.01,
                        help='If >0 enable model-free collision filtering')
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=50, help='Top-K grasps to visualize')
    parser.add_argument('--vis_top_k', type=int, default=2, help='仅可视化按 score 排序后的前 K 个抓取')
    return parser.parse_args()


def get_net(checkpoint_path, num_view):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'Loaded checkpoint: {checkpoint_path} (epoch {checkpoint.get("epoch", -1)})')
    return net, device


def depth_to_colormap(dimg):
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


def prepare_end_points(color, depth, camera_info, num_point, device, workspace_mask=None):
    # color: HxWx3 uint8 BGR; depth: HxW uint16
    h, w = depth.shape[:2]
    # create cloud (organized)
    cloud = create_point_cloud_from_depth_image(depth.astype(np.float32), camera_info, organized=True)

    # 如果没有传入 workspace_mask，默认为全黑
    if workspace_mask is None:
        workspace_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        # 保证掩码为单通道 uint8
        if workspace_mask.dtype != np.uint8:
            workspace_mask = workspace_mask.astype(np.uint8)
        # 尺寸必须匹配；若不匹配则保守地返回全黑掩码（避免误抓取）
        if workspace_mask.shape != (h, w):
            workspace_mask = np.zeros((h, w), dtype=np.uint8)

    # valid mask: workspace_mask>0 and depth>0
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    color_masked = color_rgb[mask]

    # sample
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    end_points = {}
    cloud_sampled_t = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points['point_clouds'] = cloud_sampled_t
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud, cloud_masked, color_masked


def yolo_get_mask(yolo_model, color, yolo_predict_params):
    if yolo_model is None:
        return None, None
    try:
        results = yolo_model.predict(color, **yolo_predict_params)
        r = results[0]
        seg_vis = r.plot() if hasattr(r, 'plot') else None
        if r.masks is not None and len(r.masks.data) > 0:
            mask_acc = np.zeros(color.shape[:2], dtype=np.uint8)
            for t in r.masks.data:
                m = t.detach().cpu().numpy().astype(np.uint8)
                mask_acc |= m
            if mask_acc.sum() > 0:
                return (mask_acc * 255).astype(np.uint8), seg_vis
        # 未检测到目标
        return None, seg_vis
    except Exception as e:
        print('YOLO predict failed:', e)
        return None, None


def run_graspnet_for_mask(net, device, color, depth, camera_info, args, vis, pcd, gripper_geoms, T, workspace_mask):
    # prepare inputs
    end_points, cloud, cloud_masked, color_masked = prepare_end_points(color, depth, camera_info, args.num_point, device, workspace_mask=workspace_mask)

    # update Open3D point cloud (use masked points/colors)
    update_pcd(pcd, cloud_masked, color_masked, T)
    vis.update_geometry(pcd)

    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)

    # optional collision filtering
    if args.collision_thresh > 0 and len(gg) > 0:
        mfcdetector = ModelFreeCollisionDetector(np.array(cloud[depth>0]), voxel_size=args.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=args.collision_thresh)
        gg = gg[~collision_mask]

    # postprocess：NMS + 排序
    gg.nms()
    gg.sort_by_score()

    # 垂直方向筛选（±30°）。approach 方向取旋转矩阵第一列。
    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    angle_threshold = np.deg2rad(30.0)
    keep_inds = []
    for i, grasp in enumerate(gg):
        R = grasp.rotation_matrix  # 3x3
        approach_dir = R[:, 0]
        cos_angle = float(np.dot(approach_dir, vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            keep_inds.append(i)
    if len(keep_inds) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        gg_filtered = gg
    else:
        gg_filtered = gg[keep_inds]

    # 只取垂直筛选后的前1个（已经按分数降序排序）
    if len(gg_filtered) > 1:
        gg_filtered = gg_filtered[:1]

    # 提取返回的抓取数值信息
    grasp_info = None
    if len(gg_filtered) > 0:
        g0 = gg_filtered[0]
        grasp_info = {
            'translation': g0.translation.copy(),
            'rotation_matrix': g0.rotation_matrix.copy(),
            'width': float(g0.width),
        }

    # Open3D gripper geometries and show
    grippers = gg_filtered.to_open3d_geometry_list()
    gripper_geoms = replace_grippers(vis, gripper_geoms, grippers, T)
    vis.poll_events()
    vis.update_renderer()
    return gripper_geoms, grasp_info


def main():
    args = parse_args()

    net, device = get_net(args.checkpoint_path, args.num_view)

    # 加载 YOLO 分割模型（仅当可用时），并设置目标类别
    target_class_id = 64  # 仅检测该类别（鼠标64）（香蕉46）（瓶子76）
    yolo_model = None
    if _HAS_YOLO:
        try:
            # 在 yolo11 目录或工作目录下查找权重文件
            yolo_model = YOLO(os.path.join(ROOT_DIR, 'yolo11', 'yolo11n-seg.pt'))
            yolo_predict_params = {"conf": 0.4, "iou": 0.7, "classes": [target_class_id]}
        except Exception as e:
            print('Could not load YOLO model:', e)
            yolo_model = None
    else:
        print('ultralytics YOLO not available; skipping per-object segmentation.')

    # RealSense setup（增加回退以避免 Couldn't resolve requests）
    pipeline = rs.pipeline()
    config = rs.config()
    color_w, color_h = 640, 480
    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, 30)
    align_to = rs.stream.color
    align = rs.align(align_to)
    profile = pipeline.start(config)

    # Use fixed intrinsics and factor_depth (use demo defaults)
    intrinsic = np.array([[606.44, 0.0, 322.35], [0.0, 606.48, 239.54], [0.0, 0.0, 1.0]], dtype=np.float32)
    factor_depth = float(999.999952502551)
    camera_info = CameraInfo(float(color_w), float(color_h), intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # 初始化 Open3D 可视化器（与 demo.py 相似）
    vis = o3d.visualization.Visualizer()
    # 创建窗口（宽高可根据需要调整）
    vis.create_window(window_name='GraspNet Live', width=1280, height=720)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
   
    # 为显示应用的旋转/翻转矩阵（用于调整显示方向）。
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
#   # 添加一个坐标系（XYZ）用于参考可视化 X 轴红色 Y 轴绿色 Z 轴蓝色
#     axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.0, 0.0, 0.0])
#     safe_transform(axis_frame, T)
#     vis.add_geometry(axis_frame)
    gripper_geoms = []
    # 帧计数器：用于控制算法触发频率
    frame_idx = 0

    try:
        while True:
            start_t = time.time()
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # 按帧计数进行频率限制：仅在计数能被 5 整除时运行一次 YOLO+GraspNet
            frame_idx += 1
            workspace_mask = None
            seg_vis = None
            if frame_idx % 5 == 0:
                # 使用 YOLO 得到掩码（若返回 None 则表示未检测到目标）
                workspace_mask, seg_vis = yolo_get_mask(yolo_model, color, yolo_predict_params)

                # 只有在检测到目标掩码时才运行 GraspNet；否则跳过 GraspNet 推理
                if workspace_mask is not None:
                    gripper_geoms, last_grasp_info = run_graspnet_for_mask(
                        net, device, color, depth, camera_info, args, vis, pcd, gripper_geoms, T, workspace_mask
                    )
                else:
                    # 未检测到目标时不显示任何抓手几何体（移除之前的），保持点云为深度对应的默认显示
                    if len(gripper_geoms) > 0:
                        for g in gripper_geoms:
                            try:
                                vis.remove_geometry(g)
                            except Exception:
                                pass
                        gripper_geoms = []
                    # 同时清空最近抓取信息
                    last_grasp_info = None

            # 2D 窗口：原始 RGB、对齐深度伪彩色、YOLO 分割结果、目标掩码
            depth_colormap = depth_to_colormap(depth)
            cv2.imshow('Color (raw)', color)
            if depth_colormap is not None:
                cv2.imshow('Depth (aligned)', depth_colormap)
            if seg_vis is not None:
                cv2.imshow('YOLO Segmentation', seg_vis)
            # 显示 workspace_mask（白=目标，黑=背景）
            if workspace_mask is not None:
                cv2.imshow('Workspace Mask (target)', workspace_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # save snapshot
                save_dir = os.path.join('doc', 'example_data_live')
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, 'color.png'), color)
                cv2.imwrite(os.path.join(save_dir, 'depth.png'), depth)
                try:
                    o3d_path = os.path.join(save_dir, 'o3d_grasp.png')
                    vis.capture_screen_image(o3d_path, do_render=True)
                    print(f"Saved Open3D grasp visualization to: {o3d_path}")
                except Exception as e:
                    print(f"Failed to save Open3D screenshot: {e}")
            elif key == ord('z'):
                # 打印当前抓取方向信息
                print("\n===== Current Grasp (camera frame) =====")
                if 'last_grasp_info' in locals() and last_grasp_info is not None:
                    t = last_grasp_info['translation']
                    R = last_grasp_info['rotation_matrix']
                    w = last_grasp_info['width']
                    np.set_printoptions(precision=5, suppress=True)
                    print(f"translation (m):\n{t}")
                    print(f"rotation_matrix:\n{R}")
                    print(f"width (m): {w:.5f}")
                    save_dir = os.path.join('doc', 'example_data_live')
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(save_dir, 'color.png'), color)
                    cv2.imwrite(os.path.join(save_dir, 'depth.png'), depth)
                    try:
                        o3d_path = os.path.join(save_dir, 'o3d_grasp.png')
                        vis.capture_screen_image(o3d_path, do_render=True)
                        print(f"Saved Open3D grasp visualization to: {o3d_path}")
                    except Exception as e:
                        print(f"Failed to save Open3D screenshot: {e}")
                break

            # small perf print
            t = time.time() - start_t
            # print frame time occasionally
            if t > 0:
                print(f'Frame time: {t:.3f}s, FPS: {1.0/t:.1f}', end='\r')

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
