import os
import random
import shutil
from pathlib import Path
import argparse

def split_folder_with_json(input_dir, output_base_dir, images_per_folder=100, seed=1, 
                          folder_prefix="folder", folder_names=None):
    """
    将大文件夹拆分成多个小文件夹，包含图片和对应的json文件
    
    Args:
        input_dir (str): 输入文件夹路径，包含所有要处理的文件
        output_base_dir (str): 输出基础目录，拆分后的小文件夹将创建在此目录下
        images_per_folder (int): 每个小文件夹包含的图片数量（及对应的json文件），默认100
        seed (int): 随机种子，用于控制打乱顺序的可重复性
        folder_prefix (str): 文件夹名称前缀，当没有提供具体文件夹名称时使用，默认"folder"
        folder_names (list): 可选，具体的文件夹名称列表。如果提供，将按此列表创建文件夹，
                            如果文件数量超过列表长度，剩余文件夹会按顺序追加数字
    """
    # 设置随机种子
    random.seed(seed)
    
    # 转换为Path对象
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)
    
    # 检查输入目录是否存在
    if not input_path.exists() or not input_path.is_dir():
        raise ValueError(f"输入目录不存在或不是一个有效的文件夹: {input_dir}")
    
    # 创建输出基础目录（如果不存在）
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 获取所有图片文件
    image_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("警告: 输入目录中没有找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 为每个图片文件创建一个条目，包含图片文件和对应的json文件（如果存在）
    file_groups = []
    for img_file in image_files:
        # 构建对应的json文件名
        json_file = img_file.with_suffix('.json')
        
        # 检查json文件是否存在
        has_json = json_file.exists()
        
        file_groups.append({
            'image': img_file,
            'json': json_file if has_json else None,
            'has_json': has_json
        })
    
    # 打乱文件组顺序
    # random.shuffle(file_groups)
    print("文件顺序已打乱")
    
    # 统计有json文件的数量
    json_count = sum(1 for group in file_groups if group['has_json'])
    print(f"其中 {json_count} 个图片有对应的json文件")
    
    # 计算需要创建的小文件夹数量
    num_folders = (len(file_groups) + images_per_folder - 1) // images_per_folder
    
    print(f"将创建 {num_folders} 个小文件夹，每个包含最多 {images_per_folder} 个图片文件")
    if json_count > 0:
        print(f"对应的json文件也会被移动到相应文件夹")
    
    # 准备文件夹名称
    folder_names_list = []
    if folder_names is not None:
        # 如果提供了具体的文件夹名称列表
        folder_names_list = folder_names.copy()  # 复制避免修改原始列表
        
        # 如果提供的文件夹名称不够，补充剩余的
        if len(folder_names_list) < num_folders:
            remaining = num_folders - len(folder_names_list)
            print(f"提供的文件夹名称数量 ({len(folder_names_list)}) 不足，需要 {num_folders} 个")
            print(f"将为剩余的 {remaining} 个文件夹生成名称")
            
            # 从提供的最后一个名称的序号开始继续，或者从1开始
            max_num = 0
            for name in folder_names_list:
                # 尝试从名称中提取数字
                parts = name.split('_')
                if len(parts) > 1 and parts[-1].isdigit():
                    num = int(parts[-1])
                    max_num = max(max_num, num)
            
            # 从max_num + 1开始生成剩余的名称
            for i in range(remaining):
                next_num = max_num + i + 1
                new_name = f"{folder_prefix}_{next_num:03d}"
                folder_names_list.append(new_name)
                print(f"  生成补充文件夹名称: {new_name}")
        
        # 如果提供的文件夹名称太多，只取需要的数量
        if len(folder_names_list) > num_folders:
            folder_names_list = folder_names_list[:num_folders]
            print(f"提供的文件夹名称数量 ({len(folder_names)}) 超过需要，只使用前 {num_folders} 个")
    else:
        # 没有提供具体名称，使用前缀生成
        for i in range(num_folders):
            folder_names_list.append(f"{folder_prefix}_{i + 1:03d}")
    
    print("\n将使用的文件夹名称:")
    for i, name in enumerate(folder_names_list[:min(10, len(folder_names_list))]):
        print(f"  {i + 1:3d}: {name}")
    if len(folder_names_list) > 10:
        print(f"  ... (共 {len(folder_names_list)} 个文件夹)")
    
    # 分配文件到各个小文件夹
    total_images_moved = 0
    total_jsons_moved = 0
    
    for folder_idx in range(num_folders):
        # 获取当前文件夹名称
        folder_name = folder_names_list[folder_idx]
        folder_path = output_base_path / folder_name
        folder_path.mkdir(exist_ok=True)
        
        # 计算当前文件夹的文件范围
        start_idx = folder_idx * images_per_folder
        end_idx = min(start_idx + images_per_folder, len(file_groups))
        
        # 获取当前文件夹要处理的文件组
        current_groups = file_groups[start_idx:end_idx]
        
        print(f"\n正在处理 {folder_name} (图片 {start_idx + 1} - {end_idx} / {len(file_groups)})")
        print(f"  本文件夹包含: {len(current_groups)} 个图片文件")
        
        # 统计当前文件夹的json文件数量
        current_json_count = sum(1 for group in current_groups if group['has_json'])
        if current_json_count > 0:
            print(f"  其中 {current_json_count} 个有对应的json文件")
        
        # 移动文件到小文件夹
        for group in current_groups:
            img_file = group['image']
            json_file = group['json']
            
            # 移动图片文件
            dst_img = folder_path / img_file.name
            try:
                shutil.move(str(img_file), str(dst_img))
                total_images_moved += 1
                print(f"  ✓ 已移动图片: {img_file.name}")
            except Exception as e:
                print(f"  ✗ 移动图片失败 {img_file.name}: {str(e)}")
            
            # 如果有对应的json文件，也移动它
            if group['has_json'] and json_file.exists():
                dst_json = folder_path / json_file.name
                try:
                    shutil.move(str(json_file), str(dst_json))
                    total_jsons_moved += 1
                    print(f"  ✓ 已移动JSON: {json_file.name}")
                except Exception as e:
                    print(f"  ✗ 移动JSON失败 {json_file.name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"操作完成！")
    print(f"原始图片文件总数: {len(image_files)}")
    print(f"有对应JSON文件的数量: {json_count}")
    print(f"成功移动的图片文件: {total_images_moved}")
    print(f"成功移动的JSON文件: {total_jsons_moved}")
    print(f"创建的小文件夹数量: {num_folders}")
    print(f"使用的文件夹名称前缀/列表: {folder_prefix if folder_names is None else '自定义列表'}")
    print(f"输出目录: {output_base_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    # 示例1：使用默认前缀 "folder" (folder_001, folder_002, ...)
    input_dir   = "../../data/dispose/images"
    output_dir  = "../../data/dispose/result"
    split_num   = 50
    
    # 调用方式1：使用默认命名
    split_folder_with_json(input_dir, output_dir, split_num)
    
    # 调用方式2：使用自定义前缀
    # split_folder_with_json(input_dir, output_dir, split_num, folder_prefix="batch")
    
    # 调用方式3：使用具体的文件夹名称列表
    # custom_folder_names = ["folder_31", "folder_32", "folder_33", "folder_34", "folder_35", "folder_36", "folder_37"]
    # split_folder_with_json(input_dir, output_dir, split_num, folder_names=custom_folder_names)
    
    # 调用方式4：混合使用 - 提供部分名称，剩余自动生成
    # partial_names = ["folder_057"]
    # split_folder_with_json(input_dir, output_dir, split_num, folder_names=partial_names)