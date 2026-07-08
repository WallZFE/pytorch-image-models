import os
import cv2
import glob
from tqdm import tqdm
import random

def process_images(source_folder, result_folder, num):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # 获取所有jpg和jpeg文件
    image_files = glob.glob(source_folder + "/**/*.jpg", recursive=True)
    image_files += glob.glob(source_folder + "/**/*.jpeg", recursive=True)
    image_files += glob.glob(source_folder + "/**/*.JPG", recursive=True)
    image_files += glob.glob(source_folder + "/**/*.JPEG", recursive=True)
    # 去重（防止符号链接等导致重复）
    image_files = list(set(image_files))
    
    print(f"找到 {len(image_files)} 张图片")

    random.shuffle(image_files)

    if num <= 0:
        image_files = image_files
    else:
        image_files = image_files[:num]
    
    for image_path in tqdm(image_files):
        try:
            # 获取文件名（不含路径）
            filename = os.path.basename(image_path)
            
            # 使用OpenCV读取图片（BGR格式）
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"  ✗ 无法读取 {filename}")
                continue
            
            # Resize到640x360
            resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
            
            # 计算裁剪区域：保留下面80%的部分
            # 总高度360，需要裁剪到288，所以上面要裁掉72像素
            # OpenCV中图片是numpy数组，使用数组切片裁剪: [y_start:y_end, x_start:x_end]
            cropped_img = resized_img[72:360, 0:640]
            
            # 保存到result文件夹，设置JPEG质量为95
            output_path = os.path.join(result_folder, filename)
            cv2.imwrite(output_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            print(f"  ✗ 处理 {filename} 时出错: {str(e)}")
    
    print(f"\n处理完成！结果已保存到 '{result_folder}' 文件夹")

if __name__ == "__main__":
    source_folder = "../../data/model_use/images"
    result_folder = "/data/zhifu/Project/export/XMIPC/data"
    num = 256
    process_images(source_folder, result_folder, num)