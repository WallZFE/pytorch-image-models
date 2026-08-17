import glob
import os
import cv2
import json
import numpy as np
import re
from tqdm import tqdm
import shutil

NUM_LANE = 4
NUM_ROW = 28
NUM_COL = 21
NUM_CELL_ROW = 100
NUM_CELL_COL = 100
ROW_AREA_RATIO = 0.2
LANE_LABEL_NUM = 8

# 原始字典：用于将各种同义词映射到统一的索引
LANE_LABEL_INDEX_DIC = {
    "单": 0, "单行": 0, "单列": 0,
    "多": 1, "多行": 1, "多列": 1, "双": 1, "双行": 1, "双列": 1,
    "白": 2, "白色": 2, 
    "黄": 3, "黄色": 3, 
    "实": 4, "实线": 4, "左实右实": 4, "双实线": 4, "实实线": 4,
    "虚": 5, "虚线": 5, "左虚右虚": 5, "双虚线": 5, "虚虚线": 5,
    "实虚": 6, "实虚线": 6, "左实右虚": 6, "左实右虚线": 6, 
    "虚实": 7, "虚实线": 7, "左虚右实": 7, "左虚右实线": 7,
}


# 【新增】标准名称字典：用于将索引反转回统一的中文名称，方便统计和打印
INDEX_TO_STANDARD_NAME = {
    0: "单",
    1: "多",   # "多"、"双" 等都统一显示为 "多"
    2: "白",
    3: "黄",
    4: "实线", # "实"、"双实线" 等都统一显示为 "实线"
    5: "虚线", # "虚"、"双虚线" 等都统一显示为 "虚线"
    6: "实虚线",
    7: "虚实线"
}

def get_image_files(source_root_path, image_extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']):
    image_files = list()
    for ext in image_extensions:
        image_pattern = os.path.join(source_root_path, "**", ext)
        image_files += glob.glob(image_pattern, recursive=True)
    return image_files

def process_lane_shape(shape):
    """处理单个车道形状数据, 返回处理后的字典或None"""
    try:
        if not shape.get('points') or len(shape['points']) < 2:
            print(f"跳过 shape: points 为空或数据异常")
            return None
        
        group_id = shape.get('group_id')
        if group_id is None or group_id not in [1, 2, 3, 4]:
            print(f"跳过 shape (当前group_id: {group_id}): group_id 必须在 [1, 2, 3, 4] 范围内")
            return None
        
        label_str = shape.get('label', '').strip()
        if not label_str:
            print(f"跳过 shape (group_id: {group_id}): label 为空或长度")
            return None
            
        parts = re.split(r'[,\s，\s]+', label_str)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) != 3:
            print(f"跳过 shape (group_id: {group_id}): label '{label_str}' 分割后长度为 {len(parts)},必须等于3。分割结果: {parts}")
            return None
        
        lane_label_vector = [0] * LANE_LABEL_NUM
        
        for part in parts:
            try:
                index = LANE_LABEL_INDEX_DIC[part]
                if 0 <= index < LANE_LABEL_NUM:
                    lane_label_vector[index] = 1
                else:
                    print(f"跳过 shape (group_id: {group_id}): label部分 '{part}' 对应的索引 {index} 超出范围")
                    return None
            except KeyError:
                print(f"跳过 shape (group_id: {group_id}): label部分 '{part}' 不在字典中")
                return None
        
        result = {
            'points': np.array(shape['points']),
            'label': group_id,
            'lane_label': lane_label_vector,
            'original_label': label_str,
            'label_parts': parts
        }
        return result
        
    except Exception as e:
        print(f"处理 shape 时发生错误: {str(e)}")
        return None
    
def get_json_info(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    lane_data = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'linestrip':
            processed_shape = process_lane_shape(shape)
            if processed_shape is not None:
                lane_data.append(processed_shape)
            else:
                return None
    return lane_data

def dispose_data(images_root, path_list, output_root, save_info):
    # 1. 预处理 save_info (用于复制图片)
    target_categories = []
    save_info_standard_names = set() # 记录 save_info 转换后的标准名称，用于打印时打星号
    
    if save_info:
        for info in save_info:
            folder_name = "_".join(info)
            required_indices = set()
            valid = True
            for label_part in info:
                if label_part in LANE_LABEL_INDEX_DIC:
                    required_indices.add(LANE_LABEL_INDEX_DIC[label_part])
                else:
                    print(f"Warning: save_info 中的 '{label_part}' 不在字典中")
                    valid = False
                    break
            if valid:
                target_categories.append({
                    'folder_name': folder_name,
                    'required_indices': required_indices
                })
                # 将 save_info 也转换为标准名称，方便最后对比
                standard_name = "_".join([INDEX_TO_STANDARD_NAME[idx] for idx in sorted(list(required_indices))])
                save_info_standard_names.add(standard_name)
                os.makedirs(os.path.join(output_root, folder_name), exist_ok=True)

    category_stats = {cat['folder_name']: 0 for cat in target_categories}
    
    # 【修改】用于统计所有出现的车道线类型 (使用标准名称作为 key)
    all_category_stats = {}

    # 2. 开始遍历图片
    for images_name in path_list:
        source_root_path = os.path.join(images_root, images_name)
        if not os.path.exists(source_root_path):
            print(f"Warning: {source_root_path} not exists, skipping...")
            continue
            
        image_files = get_image_files(source_root_path)
        
        for image_path in tqdm(image_files, desc=f"处理 {source_root_path}"):
            json_name = os.path.splitext(os.path.basename(image_path))[0] + '.json'
            json_path = os.path.join(os.path.dirname(image_path), json_name)

            if not os.path.exists(json_path):
                print(f"Warning: JSON file {json_path} not found, skipping...")
                continue

            json_info = get_json_info(json_path)
            if json_info is None:
                print("Warning: json path: {} is error".format(json_path))
                continue

            matched_folders_for_this_image = set()

            # 3. 遍历当前图片的每一条车道线
            for lane in json_info:
                lane_label = lane['lane_label']
                
                # 获取当前车道线激活的索引（值为1的位置）
                active_indices = set(i for i, val in enumerate(lane_label) if val == 1)
                
                # 【核心修改】将激活的索引转换为标准名称，例如 {0, 2, 5} -> "单_白_虚线"
                # 这样无论是 "单_白_虚" 还是 "单行_白色_双虚线"，都会被统一合并为 "单_白_虚线"
                standard_parts = [INDEX_TO_STANDARD_NAME[idx] for idx in sorted(list(active_indices))]
                standard_lane_type = "_".join(standard_parts)
                
                # 使用标准名称进行统计
                all_category_stats[standard_lane_type] = all_category_stats.get(standard_lane_type, 0) + 1

                # 检查这条车道线是否符合 save_info 中的某个分类 (仅当 save_info 不为空时执行)
                for cat in target_categories:
                    if cat['required_indices'].issubset(active_indices):
                        category_stats[cat['folder_name']] += 1
                        matched_folders_for_this_image.add(cat['folder_name'])

            # 4. 复制图片到匹配到的文件夹中 (仅当 save_info 不为空且匹配成功时执行)
            if matched_folders_for_this_image:
                image_basename = os.path.basename(image_path)
                json_basename = os.path.basename(json_path)

                for folder_name in matched_folders_for_this_image:
                    target_dir = os.path.join(output_root, folder_name)
                    shutil.move(image_path, os.path.join(target_dir, image_basename))
                    shutil.move(json_path, os.path.join(target_dir, json_basename))

    # 5. 打印最终统计结果
    print("\n" + "="*60)
    print(" "*20 + "车道线统计结果")
    print("="*60)
    
    if save_info:
        print("【关注的类型 (save_info)】")
        total_target_lanes = 0
        for folder_name, count in category_stats.items():
            print(f"  {folder_name:<20}: {count} 条")
            total_target_lanes += count
        print(f"  {'小计':<20}: {total_target_lanes} 条")
        print("-" * 60)

    print("【所有出现的类型 (标准名称)】")
    total_all_lanes = 0
    # 按数量降序排序
    sorted_all_stats = sorted(all_category_stats.items(), key=lambda x: x[1], reverse=True)
    for lane_type, count in sorted_all_stats:
        # 如果这个标准类型刚好也在 save_info 中，打个星号标记
        mark = " *" if lane_type in save_info_standard_names else ""
        print(f"  {lane_type:<20}: {count} 条{mark}")
        total_all_lanes += count
        
    print("-" * 60)
    print(f"  {'总计':<20}: {total_all_lanes} 条")
    print("="*60 + "\n")

def main(images_root, train_list, output_root, save_info):
    dispose_data(images_root, train_list, output_root, save_info)

if __name__ == "__main__":
    images_root = '../../data/model_use'
    train_list  = ["20260417", "20260421", "20260429", "20260520", "20260525", "20260527", "20260603", "20260617", "20260630", "images"]
    # train_list  = ["images_test"]
    output_root = '../../data/dispose/result'
    
    # 测试时可以把 save_info 设为 []，只统计不复制
    # save_info = [["单","白","虚实线"], ["多","白","实线"], ["多","黄","虚线"], ["多","黄","实线"]]
    save_info = [] 
    
    os.makedirs(output_root, exist_ok=True)
    main(images_root, train_list, output_root, save_info)