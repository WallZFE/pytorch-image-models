import os
import cv2
import numpy as np
from PIL import Image
import glob

def create_video_from_images(folder_path, output_video_path, fps=30):
    """
    读取文件夹下所有图片，按名字排序，生成视频
    
    参数:
    folder_path: 图片文件夹路径
    output_video_path: 输出视频路径
    fps: 视频帧率
    """
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
    
    # 按文件名排序
    image_files.sort(key=lambda x: os.path.basename(x))
    
    if not image_files:
        print("没有找到图片文件！")
        return
    
    print(f"找到 {len(image_files)} 张图片:")
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"无法读取第一张图片: {image_files[0]}")
        return
    
    height, width = first_image.shape[:2]
    print(f"视频尺寸: {width}x{height}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("无法创建视频写入器！")
        return
    
    # 处理每张图片
    for i, image_file in enumerate(image_files):
        print(f"处理图片 {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # 读取图片
        img = cv2.imread(image_file)
        if img is None:
            print(f"  跳过无法读取的图片: {image_file}")
            continue
        
        # 调整图片尺寸到统一大小
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
    
    # 释放资源
    video_writer.release()
    print(f"视频生成成功: {output_video_path}")
    print(f"视频信息: {width}x{height}, {fps}fps")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_folder = "/data/zhifu/Project/deploy/XMIPC/adas_shangqi_liu/XMIPCLinuxV100R003C00SPC030/sample/npu/xmm/result"  # 图片文件夹路径
    output_video = "output_video.mp4"  # 输出视频路径
    
    # 生成视频
    create_video_from_images(
        folder_path=input_folder,
        output_video_path=output_video,
        fps=10,           # 帧率
    )
    
    print("\n完成！")