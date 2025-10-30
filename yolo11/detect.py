from ultralytics import YOLO
import cv2
import numpy as np

# 1. 加载分割模型
model = YOLO("yolo11n-seg.pt")

# 2. 配置预测参数（重点：指定目标类别ID）
target_class_id = 46  # 例如：46对应"banana"，根据你的目标修改
predict_params = {
    "classes": [target_class_id],  # 仅检测该ID对应的类别
    "conf": 0.5,  # 置信度阈值（过滤低置信度结果，0~1）
    "iou": 0.7,   # NMS交并比阈值（去除重叠框）
    # "save": True  # 是否保存结果（按需开启）
}

# 3. 处理图片并获取目标物体的结果
image_path = "/home/jyx/graspnet/doc/example_data/color.png"
results = model.predict(image_path, **predict_params)
results[0].save()

# 指定保存路径和文件名（推荐，方便查找）
save_path = "output_segment.jpg"  # 可自定义路径，如"./results/my_result.jpg"
results[0].save(save_path)

if results[0].masks is not None:  # 确保检测到目标，否则masks为None
    # 遍历每个目标的掩码（如果有多个目标）
    for i in range(len(results[0].masks)):
        # 4.1 提取单个目标的掩码（numpy数组，形状：(H, W)，值为0或1）
        mask = results[0].masks.data[i].cpu().numpy()  # 先转CPU再转numpy，0=背景，1=目标

        # 4.2 转换为8位黑白图（0=黑，255=白，方便保存和查看）
        mask_8bit = (mask * 255).astype(np.uint8)  # 0→0（黑），1→255（白）
        
        # 4.3 保存纯掩码图片（独立文件）
        mask_save_path = f"mask_target_{i}.jpg"  # 文件名：mask_target_0.jpg, 1.jpg...
        cv2.imwrite(mask_save_path, mask_8bit)
        print(f"第{i+1}个目标的掩码已保存到：{mask_save_path}")
        
        # 4.4 （可选）显示掩码图片
        cv2.imshow(f"Mask of Target {i+1}", mask_8bit)
        cv2.waitKey(0)  # 按任意键关闭窗口
    cv2.destroyAllWindows()
else:
    print("未检测到目标，无掩码结果")
