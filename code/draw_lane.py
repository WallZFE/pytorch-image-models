#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py
功能：在图片/YUV/视频的某一个高度处画一条红线

用法：
    python demo.py <输入文件路径>

支持的输入格式：
    - 图片: .jpg, .jpeg, .png, .bmp, .tif, .tiff
    - 视频: .mp4, .avi, .mkv, .mov, .flv
    - YUV:  .yuv  (NV21: 420sp, Y平面 + VUVUVU交错排列)

示例：
    python demo.py image.jpg
    python demo.py video.mp4
    python demo.py test.yuv
"""

import sys
import os
import re
import argparse
import cv2
import numpy as np
from datetime import datetime

LANE_NUM = 65


# ==================== 支持的扩展名 ====================
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.ts'}
YUV_EXTS   = {'.yuv'}


def get_timestamp():
    """返回 HHMMSS 格式的时间戳"""
    return datetime.now().strftime("%H%M%S")

def get_file_type(filepath):
    """根据扩展名判断文件类型"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in IMAGE_EXTS:
        return 'image'
    elif ext in VIDEO_EXTS:
        return 'video'
    elif ext in YUV_EXTS:
        return 'yuv'
    else:
        return 'unknown'

def read_yuv_nv21(filepath, width, height):
    """
    读取 NV21 格式的 YUV 文件 (420sp, Y + VUVUVU交错)
    NV21 内存布局：
      - Y 平面: width * height 字节
      - UV 交错平面: width * height / 2 字节 (V在前, U在后, 即 VUVUVU...)
      - 总计: width * height * 3 / 2 字节
    返回 BGR 图像 (numpy array)
    """
    frame_size = width * height * 3 // 2
    file_size = os.path.getsize(filepath)

    if file_size < frame_size:
        print(f"[错误] YUV文件太小: {file_size} 字节, "
              f"预期至少 {frame_size} 字节 (分辨率 {width}x{height})")
        sys.exit(1)

    with open(filepath, 'rb') as f:
        raw = f.read(frame_size)

    yuv_data = np.frombuffer(raw, dtype=np.uint8)

    # OpenCV 的 cvtColor COLOR_YUV2BGR_NV21 要求输入 shape 为 (height * 3 // 2, width)
    yuv_image = yuv_data.reshape((height * 3 // 2, width))
    bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)

    return bgr_image


def draw_line(frame):
    """
    在图像高度处画一条红色横线
    """
    h, w = frame.shape[:2]
    line_y = int(h * (1 - LANE_NUM / 100.0))

    # ---------- 画红色线 ----------
    cv2.line(frame, (0, line_y), (w-1, line_y), (0, 0, 255), 1)

    # ---------- 准备文字 ----------
    text = f"{LANE_NUM}%"

    # 调整文字位置，避免文字被截断
    text_y = line_y - 10
    if text_y < 20:  # 如果太靠近顶部，放在线下方
        text_y = line_y + 20
    
    text_position = (10, text_y)  # 文字位置
    
    # 绘制文字背景（可选，提高可读性）
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 8)[0]
    text_bg_top_left = (text_position[0] - 2, text_position[1] - text_size[1] - 2)
    text_bg_bottom_right = (text_position[0] + text_size[0] + 2, text_position[1] + 2)
    cv2.rectangle(frame, text_bg_top_left, text_bg_bottom_right, (0, 0, 0), -1)
    
    # 绘制文字
    cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)

    return frame


def process_image(filepath):
    """处理图片文件"""
    frame = cv2.imread(filepath)
    if frame is None:
        print(f"[错误] 无法读取图片: {filepath}")
        sys.exit(1)

    frame = draw_line(frame)

    output_name = f"output_{get_timestamp()}.jpg"
    cv2.imwrite(output_name, frame)
    print(f"[完成] 已保存: {output_name}")


def process_yuv(filepath):
    """处理 YUV (NV21) 文件"""
    frame = read_yuv_nv21(filepath, 3840, 2160)
    frame = draw_line(frame)

    output_name = f"output_{get_timestamp()}.jpg"
    cv2.imwrite(output_name, frame)
    print(f"[完成] 已保存: {output_name}")


def process_video(filepath):
    """处理视频文件：逐帧处理并输出视频"""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {filepath}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[信息] 视频: {width}x{height}, {fps:.2f}fps, {total_frames} 帧")

    output_name = f"output_{get_timestamp()}.mp4"

    # 使用 mp4v 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频: {output_name}")
        cap.release()
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = draw_line(frame)
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"  处理进度: {frame_idx}/{total_frames} "
                  f"({frame_idx * 100 // total_frames}%)")

    cap.release()
    writer.release()
    print(f"[完成] 已保存: {output_name}  (共 {frame_idx} 帧)")


def main():
    parser = argparse.ArgumentParser(
        description=f"在图片/YUV/视频的高度{LANE_NUM}%处画红线并标注",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                示例:
                python demo.py image.jpg
                python demo.py video.mp4
                python demo.py test.yuv
            """
    )
    parser.add_argument('input_file', type=str, help='输入文件路径(图片/视频/YUV)')

    args = parser.parse_args()

    filepath = args.input_file

    if not os.path.isfile(filepath):
        print(f"[错误] 文件不存在: {filepath}")
        sys.exit(1)

    file_type = get_file_type(filepath)
    print(f"[信息] 输入文件: {filepath}")
    print(f"[信息] 文件类型: {file_type}")

    if file_type == 'image':
        process_image(filepath)
    elif file_type == 'video':
        process_video(filepath)
    elif file_type == 'yuv':
        process_yuv(filepath)
    else:
        print(f"[错误] 不支持的文件格式: {os.path.splitext(filepath)[1]}")
        print(f"  支持的图片格式: {', '.join(sorted(IMAGE_EXTS))}")
        print(f"  支持的视频格式: {', '.join(sorted(VIDEO_EXTS))}")
        print(f"  支持的YUV格式:  {', '.join(sorted(YUV_EXTS))}")
        sys.exit(1)

if __name__ == '__main__':
    main()
