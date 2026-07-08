import glob
import os
import cv2
import json
import numpy as np
import re
from tqdm import tqdm
from scipy.interpolate import interp1d
import shutil

TEST_MODE = False

NUM_LANE = 4
NUM_ROW = 58
NUM_COL = 65
NUM_CELL_ROW = 100
NUM_CELL_COL = 100
ROW_AREA_RATIO = 0.2
LANE_LABEL_NUM = 8
LANE_LABEL_INDEX_DIC= {"单": 0, "单行": 0, "单列": 0,
                        "多": 1, "多行": 1, "多列": 1, "双": 1, "双行": 1, "双列": 1,
                        "白": 2, "白色": 2, 
                        "黄": 3, "黄色": 3, 
                        "实": 4, "实线": 4, "左实右实": 4, "双实线": 4, "实实线": 4,
                        "虚": 5, "虚线": 5, "左虚右虚": 5, "双虚线": 5, "虚虚线": 5,
                        "实虚": 6, "实虚线": 6, "左实右虚": 6, "左实右虚线": 6, 
                        "虚实": 7, "虚实线": 7, "左虚右实": 7, "左虚右实线": 7,
                    }
# 锚点设置
ROW_COORDS = [115, 118, 120, 123, 125, 128, 130, 133, 136, 139,
              141, 144, 147, 150, 153, 156, 160, 163, 166, 170,
              173, 177, 181, 184, 188, 192, 197, 201, 205, 209,
              214, 218, 223, 228, 233, 238, 243, 248, 253, 258,
              263, 269, 274, 280, 285, 291, 297, 302, 308, 314,
              320, 326, 333, 339, 345, 350, 355, 359]

COL_COORDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 
              100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
              200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
              300, 310, 320, 330, 340, 350, 360, 370, 380, 390,
              400, 410, 420, 430, 440, 450, 460, 470, 480, 490,
              500, 510, 520, 530, 540, 550, 560, 570, 580, 590,
              600, 610, 620, 630, 639]

# ROW_COORDS = [130, 140, 150, 157, 160, 164, 167, 170, 173, 176,
#             180, 187, 195, 205, 215, 225, 235, 245, 255, 265,
#             275, 295, 315, 335, 355]
# COL_COORDS = [10, 60, 110, 160, 210, 230, 250, 270, 290, 310,
#             330, 350, 370, 390, 410, 430, 480, 530, 580, 630]

def calc_k(line):
    '''
    Calculate the direction of lanes
    '''
    line_x = line[:, 0]
    line_y = line[:, 1]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 90:
        return -10  # if the lane is too short, it will be skipped
    
    p = np.polyfit(line_x, line_y, deg=1)
    rad = np.arctan(p[0])

    return rad

def get_image_files(source_root_path,  image_extensions=['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']):
    image_files = list()
    for ext in image_extensions:
        image_pattern = os.path.join(source_root_path, "**", ext)
        image_files += glob.glob(image_pattern, recursive=True)
    return image_files

def process_lane_shape(shape):
    """
    处理单个车道形状数据, 返回处理后的字典或None
    
    Args:
        shape: 包含车道信息的字典
        
    Returns:
        dict: 处理后的字典, 包含points、label、lane_label
        None: 如果不符合条件需要跳过
    """
    try:
        # 检查points是否有值
        if not shape.get('points') or len(shape['points']) < 2:
            print(f"跳过 shape: points 为空或数据异常")
            return None
        
        # 检查group_id是否有值且在1-4范围内
        group_id = shape.get('group_id')
        if group_id is None or group_id not in [1, 2, 3, 4]:
            print(f"跳过 shape (当前group_id: {group_id}): group_id 必须在 [1, 2, 3, 4] 范围内")
            return None
        
        # 3. 检查label，按空格或逗号分割，必须等于3部分
        label_str = shape.get('label', '').strip()
        if not label_str:
            print(f"跳过 shape (group_id: {group_id}): label 为空或长度")
            return None
        # 使用正则表达式分割，支持中文逗号、英文逗号、空格、中文空格
        parts = re.split(r'[,\s，\s]+', label_str)
        parts = [part.strip() for part in parts if part.strip()]  # 过滤空字符串并清理
        if len(parts) != 3:
            print(f"跳过 shape (group_id: {group_id}): label '{label_str}' 分割后长度为 {len(parts)},必须等于3。分割结果: {parts}")
            return None
        
        # 4. 生成lane_label向量 (8个0)
        lane_label_vector = [0] * LANE_LABEL_NUM
        
        # 5. 根据分割后的parts设置对应位置为1
        for part in parts:
            try:
                index = LANE_LABEL_INDEX_DIC[part]
                if 0 <= index < LANE_LABEL_NUM:
                    lane_label_vector[index] = 1
                else:
                    print(f"跳过 shape (group_id: {group_id}): label部分 '{part}' 对应的索引 {index} 超出范围 [0, {LANE_LABEL_NUM-1}]")
                    return None
            except KeyError:
                print(f"跳过 shape (group_id: {group_id}): label部分 '{part}' 不在 LANE_LABEL_INDEX_DIC 中。可用键: {list(LANE_LABEL_INDEX_DIC.keys())[:10]}...")
                return None
        
        # 6. 构建返回的字典
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
        print(f"有问题的 shape 数据: {shape}")
        return None

def validate_info(lane_intersections):
    '''
    检查数据是否符合要求
    '''
    
    # 1. 判断lane_intersections有值
    if not lane_intersections:
        print("WARNING: lane_intersections为空, 没有车道线数据")
        return False
    
    # 2. 检查每个元素的points列表至少有两个点
    for i, lane in enumerate(lane_intersections):
        if 'points' not in lane or not isinstance(lane['points'], list):
            print(f"WARNING: 第{i+1}条车道线缺少points字段或格式不正确")
            return False
        if len(lane['points']) < 2:
            print(f"WARNING: 第{i+1}条车道线的points列表少于2个点, 当前点数: {len(lane['points'])}")
            return False
    
    # 3. 检查label值只能是1,2,3,4且不能重复
    labels = []
    for i, lane in enumerate(lane_intersections):
        if 'label' not in lane:
            print(f"WARNING: 第{i+1}条车道线缺少label字段")
            return False
        
        try:
            label_val = int(lane['label'])
        except (ValueError, TypeError):
            print(f"WARNING: 第{i+1}条车道线的label值不是有效数字: {lane['label']}")
            return False
        
        if label_val not in [1, 2, 3, 4]:
            print(f"WARNING: 第{i+1}条车道线的label值不在1-4范围内: {label_val}")
            return False
        
        labels.append(label_val)
    
    if len(labels) != len(set(labels)):
        print(f"WARNING: label值有重复。当前labels: {labels}")
        return False
    
    # 4. 检查lane_label是8个值，必须是三个1，五个0
    for i, lane in enumerate(lane_intersections):
        if 'lane_label' not in lane:
            print(f"WARNING: 第{i+1}条车道线缺少lane_label字段")
            return False
        
        if not isinstance(lane['lane_label'], list):
            print(f"WARNING: 第{i+1}条车道线的lane_label不是列表类型")
            return False
        
        if len(lane['lane_label']) != 8:
            print(f"WARNING: 第{i+1}条车道线的lane_label长度不是8,当前长度: {len(lane['lane_label'])}")
            return False
        
        count_ones = sum(1 for x in lane['lane_label'] if x == 1)
        count_zeros = sum(1 for x in lane['lane_label'] if x == 0)
        
        if count_ones != 3 or count_zeros != 5:
            print(f"WARNING: 第{i+1}条车道线的lane_label不符合要求(需要3个1和5个0).当前: {lane['lane_label']}, 1的个数: {count_ones}, 0的个数: {count_zeros}")
            return False
    
    # 5. 根据label排序并检查x值顺序
    # 先按label排序
    sorted_lanes = sorted(lane_intersections, key=lambda x: int(x['label']))
    
    # 为每条线创建y->x的映射
    lane_x_values = {}
    for lane in sorted_lanes:
        label = int(lane['label'])
        points = lane['points']
        
        # 创建点映射，便于查找
        point_dict = {}
        for x, y in points:
            # 找到最接近的ROW_COORDS值
            closest_y = min(ROW_COORDS, key=lambda ry: abs(ry - y))
            if abs(closest_y - y) < 10:  # 允许一定误差
                if closest_y not in point_dict or abs(y - closest_y) < abs(point_dict[closest_y][1] - closest_y):
                    point_dict[closest_y] = (x, y)
        
        lane_x_values[label] = point_dict
    
    # 检查每条线是否都有足够的共同y值
    common_ys = set(ROW_COORDS)
    for label, points_dict in lane_x_values.items():
        common_ys = common_ys.intersection(set(points_dict.keys()))
    
    # 至少需要2个共同y值进行比较
    if len(common_ys) >= 2:
        # 检查在共同y值下，x值是否按label顺序递增
        for y in sorted(common_ys):
            x_values = []
            labels_present = []
            
            for label in sorted(lane_x_values.keys()):
                if y in lane_x_values[label]:
                    x_val = lane_x_values[label][y][0]
                    x_values.append(x_val)
                    labels_present.append(label)
            
            if len(x_values) >= 2:  # 至少有两条线在这个y值上有数据
                # 检查x值是否按label顺序递增
                for i in range(1, len(x_values)):
                    if x_values[i] <= x_values[i-1]:
                        print(f"WARNING: 在y={y}处，label={labels_present[i]}的x值({x_values[i]})不大于label={labels_present[i-1]}的x值({x_values[i-1]})")
                        print(f"WARNING: 所有x值: {list(zip(labels_present, x_values))}")
                        return False
    
    return True
    
def get_json_info(json_path):
    # 加载JSON标注数据
    with open(json_path) as f:
        data = json.load(f)
    
    # 提取车道线点和标签
    lane_data = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'linestrip':
            processed_shape = process_lane_shape(shape)
            if processed_shape is not None:
                lane_data.append(processed_shape)
            else:
                return None
    return lane_data

def find_intersection(p1, p2, y):
    """
    计算交点
    """
    x1, y1 = p1
    x2, y2 = p2

    if (y1 <= y <= y2) or (y2 <= y <= y1):
        if y1 == y2:
            return None
        t = (y - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return (int(x), int(y))
    return None

def convert_to_json(image_path, infos, output_json):
    data = {"lanes": [], "lane_label": [], "lane_index":[], "h_samples": ROW_COORDS, "raw_file": image_path}

    with open(output_json, 'a') as out_f:  # 追加模式
        for info in infos:       
            data['lane_index'].append([info['label']])
            data['lane_label'].append(info['lane_label'])

            # 转换为数值并转为(y, x)格式
            points = [(int(y), int(x)) for x, y in info['points']]

            # 创建字典方便查找（y -> x）
            point_dict = {y: x for y, x in points}

            lane = []
            for y in ROW_COORDS:
                lane.append(point_dict.get(y, -2))  # 存在y则取x，否则-2

            data['lanes'].append(lane)
            
        # 写入一行JSON，不换行
        json_line = json.dumps(data, separators=(', ', ': '))
        out_f.write(json_line + '\n')

def draw(im, line, idx):
    # # 调整线宽范围（整体加粗）
    # min_width = 5  # 最细处（远处）
    # max_width = 50  # 最粗处（近处）

    # # 计算每个点的y坐标（越大表示越近）
    # y_coords = line[:, 1]

    # # 归一化到0-1范围（1表示最近处）
    # norm_depths = (y_coords - min(y_coords)) / (max(y_coords) - min(y_coords))

    # # 计算每个点的动态线宽（近粗远细）
    # widths = min_width + (max_width - min_width) * norm_depths

    # # 插值生成平滑线宽变化
    # x = np.arange(len(line))
    # width_interp = interp1d(x, widths, kind='linear', fill_value='extrapolate')

    # # 绘制线段（带平滑过渡）
    # for i in range(len(line) - 1):
    #     pt1 = tuple(line[i])
    #     pt2 = tuple(line[i + 1])

    #     # 当前线段的平均线宽（取整）
    #     current_width = int((width_interp(i) + width_interp(i + 1)) / 2)

    #     # 绘制线段（使用车道ID作为像素值）
    #     cv2.line(im, pt1, pt2, color=idx, thickness=current_width)

    #     # 在线段连接处绘制圆点消除接缝
    #     cv2.circle(im, pt1, current_width // 2, idx, -1)
    # # 后处理：膨胀操作确保连续性
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)

    line_x = line[:, 0]
    line_y = line[:, 1]
    pt0 = (int(line_x[0]), int(line_y[0]))

    if TEST_MODE:
        cv2.putText(im, str(idx), (int(line_x[len(line_x) // 2]), int(line_y[len(line_x) // 2]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        idx = min(255, max(0, idx * 60))
    else:
        idx = idx

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1])), (idx,), thickness=16)
        pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))

def generate_segmentation_and_list(image_path, lane_intersections, image_shape, save_mask_path):
    lines       = [np.array(l['points'], dtype=np.float64) for l in lane_intersections]
    lane_label  = [np.array(l['lane_label']) for l in lane_intersections]

    # 车道线序号 没有用到 做参考作用
    lane_index  = [l['label'] for l in lane_intersections]

    ''' 
    左侧车道线 值为负值 值越小越靠近中间
    右侧车道线 值为正值 越大越靠近中间
    '''
    ks = np.array([calc_k(line) for line in lines])  # get the direction of each lane

    # 左侧车道线斜率值
    k_neg = ks[ks < 0].copy()
    # 右侧车道线斜率值
    k_pos = ks[ks > 0].copy()

    k_neg = k_neg[k_neg != -10]  # -10 means the lane is too short and is discarded
    k_pos = k_pos[k_pos != -10]

    # 递增排序
    k_neg.sort()
    k_pos.sort()

    # height, width
    label = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)

    # 车道线点
    all_points = np.zeros((NUM_LANE, NUM_ROW, 2), dtype=np.float64)
    all_points[:, :, 1] = np.tile(ROW_COORDS, (NUM_LANE, 1))
    all_points[:, :, 0] = -99999

    # 车道线类型
    bin_lane_label = np.zeros((NUM_LANE, LANE_LABEL_NUM), dtype=np.uint8)

    # 车道线是否存在
    bin_label = [0, 0, 0, 0]
    
    # 处理左侧车道
    if len(k_neg) == 1:  # for only one lane in the left
        # 中间左侧车道线
        which_lane = np.where(ks == k_neg[0])[0][0]
        draw(label, lines[which_lane], 2)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[1, yy, 0] = xx
        bin_label[1] = 1
        bin_lane_label[1] = lane_label[which_lane]
    elif len(k_neg) >= 2:  # for more than two lanes in the left
        # 最外侧左车道线
        which_lane = np.where(ks == k_neg[1])[0][0]  # we only choose the two lanes that are closest to the center
        draw(label, lines[which_lane], 1)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[0, yy, 0] = xx
        bin_label[0] = 1
        bin_lane_label[0] = lane_label[which_lane]

        # 中间左侧车道线
        which_lane = np.where(ks == k_neg[0])[0][0]
        draw(label, lines[which_lane], 2)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[1, yy, 0] = xx
        bin_label[1] = 1
        bin_lane_label[1] = lane_label[which_lane]

    # 处理右侧车道
    if len(k_pos) == 1:  # For the lanes in the right, the same logical is adopted.
        # 中间右侧车道线
        which_lane = np.where(ks == k_pos[0])[0][0]
        draw(label, lines[which_lane], 3)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[2, yy, 0] = xx
        bin_label[2] = 1
        bin_lane_label[2] = lane_label[which_lane]
    elif len(k_pos) >= 2:
        # 中间右侧车道线
        which_lane = np.where(ks == k_pos[-1])[0][0]
        draw(label, lines[which_lane], 3)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[2, yy, 0] = xx
        bin_label[2] = 1
        bin_lane_label[2] = lane_label[which_lane]

        # 最外侧右车道线
        which_lane = np.where(ks == k_pos[-2])[0][0]
        draw(label, lines[which_lane], 4)
        xx = np.array(lines[which_lane][:, 0])
        yy = []
        for y in lines[which_lane][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[3, yy, 0] = xx
        bin_label[3] = 1
        bin_lane_label[3] = lane_label[which_lane]

    cv2.imwrite(save_mask_path, label)

    cache_dict = {}
    cache_dict[image_path] = {
            "points": all_points.tolist(),
            "lane_label": bin_lane_label.tolist()
        }
    
    bin_lane_label_str = ''
    for row in bin_lane_label:
        bin_lane_label_str += ' ' + ' '.join(map(str, row))
    
    return cache_dict, bin_label, bin_lane_label_str

def generate_segmentation_and_list2(image_path, lane_intersections, image_shape, save_mask_path):
    lines       = [np.array(l['points'], dtype=np.float64) for l in lane_intersections]
    lane_label  = [np.array(l['lane_label']) for l in lane_intersections]

    lane_index  = [int(l['label']) for l in lane_intersections]

    # height, width
    label = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)

    # 车道线点
    all_points = np.zeros((NUM_LANE, NUM_ROW, 2), dtype=np.float64)
    all_points[:, :, 1] = np.tile(ROW_COORDS, (NUM_LANE, 1))
    all_points[:, :, 0] = -99999

    # 车道线类型
    bin_lane_label = np.zeros((NUM_LANE, LANE_LABEL_NUM), dtype=np.uint8)

    # 车道线是否存在
    bin_label = [0, 0, 0, 0]

    for i, lane_idx in enumerate(lane_index):
        if lane_idx < 1 or lane_idx > NUM_LANE:
            continue
        draw(label, lines[i], lane_idx)
        xx = np.array(lines[i][:, 0])
        yy = []
        for y in lines[i][:, 1]:
            distances = np.abs(ROW_COORDS - y)
            idx = np.argmin(distances).astype(int)
            yy.append(idx)
        all_points[lane_idx-1, yy, 0] = xx
        bin_label[lane_idx-1] = 1
        bin_lane_label[lane_idx-1] = lane_label[i]

    cv2.imwrite(save_mask_path, label)

    cache_dict = {}
    cache_dict[image_path] = {
            "points": all_points.tolist(),
            "lane_label": bin_lane_label.tolist()
        }
    
    bin_lane_label_str = ''
    for row in bin_lane_label:
        bin_lane_label_str += ' ' + ' '.join(map(str, row))
    
    return cache_dict, bin_label, bin_lane_label_str

def draw_and_save_lanes(original_img: np.ndarray,  img_path: str,  lane_points: list,  row_coords: list,  output_root: str,  max_w: int = 3840,  max_h: int = 2160):
    """
    在图像上绘制车道线点、线、序号及水平参考线，并等比缩放后保存。
    
    Args:
        original_img (np.ndarray): 原始图像 (H, W, C)
        img_path (str): 图像相对路径，用于构建保存目录和文件名
        lane_points (list): 车道线点列表，即原 temp_cache_dict[img_path]['points']
        row_coords (list): 水平参考线的 Y 坐标列表 (原始分辨率下的坐标)
        output_root (str): 输出根目录
        max_w (int): 目标最大宽度 (默认 1920)
        max_h (int): 目标最大高度 (默认 1080)
    """
    # ================= 1. 图像等比缩放 =================
    orig_h, orig_w = original_img.shape[:2]
    scale = min(max_w / orig_w, max_h / orig_h)
    
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # 使用 INTER_AREA 缩小图片抗锯齿效果最好
    img_canvas = cv2.resize(original_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # ================= 2. 构建保存路径 =================
    result_dir = os.path.join(output_root, 'result_img_print')
    os.makedirs(result_dir, exist_ok=True)
    result_subdir = os.path.join(result_dir, os.path.dirname(img_path))
    os.makedirs(result_subdir, exist_ok=True)
    result_img_path = os.path.join(result_subdir, os.path.basename(img_path))

    # ================= 3. 绘制车道线元素 =================
    for idx, lane_line in enumerate(lane_points):
        valid_points = []
        for point in lane_line:
            x, y = point
            if x != -99999 and y != -99999:  # 过滤无效坐标
                # 坐标乘上缩放比例
                valid_points.append([int(x * scale), int(y * scale)])
        
        if len(valid_points) == 0:
            continue
        
        points = np.array(valid_points, dtype=np.int32)
        
        # 3.1 绘制点 (半径随缩放比例自适应)
        point_radius = max(1.5, int(3 * scale))
        for point in points:
            cv2.circle(img_canvas, (point[0], point[1]), point_radius, (0, 0, 255), -1)

        # # 3.2 绘制首尾坐标文本 (显示原始坐标，方便对照数据)
        # # 获取原始坐标
        # orig_x0, orig_y0 = int(lane_line[0][0]), int(lane_line[0][1])
        # orig_x1, orig_y1 = int(lane_line[-1][0]), int(lane_line[-1][1])
        
        # font_scale_text = 0.5 * scale + 0.2
        # y_offset = int(20 * scale) * idx
        
        # cv2.putText(img_canvas, f"({orig_x0},{orig_y0})", 
        #             (points[0][0] + int(5*scale), points[0][1] - int(5*scale) + y_offset), 
        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (255, 255, 255), 1)
                    
        # cv2.putText(img_canvas, f"({orig_x1},{orig_y1})", 
        #             (points[-1][0] + int(5*scale), points[-1][1] - int(5*scale) - y_offset), 
        #             cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (255, 255, 255), 1)
        
        # 3.3 绘制车道线
        if len(points) > 1:
            line_thickness = max(1, int(2 * scale))
            cv2.polylines(img_canvas, [points.reshape(-1, 1, 2)], False, (0, 255, 0), line_thickness)
        
        # 3.4 在车道线中间位置标注序号
        if len(points) >= 2:
            mid_point = np.mean(points, axis=0).astype(np.int32)
            
            if 0 <= mid_point[0] < img_canvas.shape[1] and 0 <= mid_point[1] < img_canvas.shape[0]:
                text = f"Lane {idx+1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale_label = 0.7 * scale + 0.2
                thickness_label = max(1, int(2 * scale))
                color = (255, 255, 255)  
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale_label, thickness_label)
                
                bg_top_left = (mid_point[0] - text_width//2 - 5, mid_point[1] - text_height//2 - 5)
                bg_bottom_right = (mid_point[0] + text_width//2 + 5, mid_point[1] + text_height//2 + 5)
                cv2.rectangle(img_canvas, bg_top_left, bg_bottom_right, (0, 0, 0), -1)  
                
                cv2.putText(img_canvas, text, (mid_point[0] - text_width//2, mid_point[1] + text_height//2), 
                        font, font_scale_label, color, thickness_label)

    # # ================= 4. 绘制水平参考线 =================
    # for num, given_y in enumerate(row_coords):
    #     scaled_y = int(given_y * scale)
        
    #     if 0 <= scaled_y < new_h:
    #         line_color = (0, 0, 255)  
    #         line_thickness = max(1, int(2 * scale)) 

    #         # 使用缩放后的 new_w 作为右侧边界
    #         cv2.line(img_canvas, (0, scaled_y), (new_w - 1, scaled_y), line_color, line_thickness)
            
    #         line_text = f"{num}"
    #         text_position = (10 + num * 10, scaled_y) 
    #         if text_position[1] > 0: 
    #             cv2.putText(img_canvas, line_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ================= 5. 保存图像 =================
    cv2.imwrite(result_img_path, img_canvas)

def dispose_data(images_root, path_list, output_root, resize_size, mode='train'):
    cache_dict = dict()
    error_root = os.path.join(output_root, 'error_result')
    os.makedirs(error_root, exist_ok=True)

    # 统计坐标的车道线数量分布
    # y_row_counts[y] = {0: 出现0条线的图片数, 1: 出现1条线的图片数, ..., 4: 出现4条线的图片数}
    y_row_counts = {y: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} for y in ROW_COORDS}
    y_row_total = {y: 0 for y in ROW_COORDS}  # 每个y坐标上所有图片的总线数
    y_row_img_count = {y: 0 for y in ROW_COORDS}  # 每个y坐标有车道线的图片数
    total_images = 0

    if mode == 'train':
        print("=== 处理训练数据 ===")
        label_file                  = os.path.join(output_root, 'train_label.json')
        gt_file                     = os.path.join(output_root, 'train_gt.txt')
        tusimple_anno_cache_file    = os.path.join(output_root, 'tusimple_anno_cache.json')
    elif mode == 'test':
        print("=== 处理测试数据 ===")
        label_file                  = os.path.join(output_root, 'test_label.json')
        gt_file                     = os.path.join(output_root, 'test.txt')
        tusimple_anno_cache_file    = os.path.join(output_root, 'tusimple_anno_cache_test.json')
    elif mode == 'error_train':
        print("=== 处理背景图数据 ===")
        label_file                  = os.path.join(output_root, 'train_label.json')
        gt_file                     = os.path.join(output_root, 'train_gt.txt')
        tusimple_anno_cache_file    = os.path.join(output_root, 'tusimple_anno_cache.json')
    else:
        print("=== mode error ===")
        return 
    
    if os.path.exists(tusimple_anno_cache_file):
        with open(tusimple_anno_cache_file, 'r') as f:
            cache_dict.update(json.load(f))

    for images_name in path_list:
        source_root_path = os.path.join(images_root, images_name)
        if not os.path.exists(source_root_path):
            print(f"Warning: {source_root_path} not exists, skipping...")
            continue
        image_files = get_image_files(source_root_path)
        
        # 处理每张图片
        for image_path in tqdm(image_files, desc=f"处理 {source_root_path}"):
            json_name = os.path.splitext(os.path.basename(image_path))[0] + '.json'
            json_path = os.path.join(os.path.dirname(image_path), json_name)

            # 加载原始图像
            original_img = cv2.imread(image_path)
            original_img_h, original_img_w,_ = original_img.shape
            if original_img is None:
                print("Warning: image path: {} is error".format(image_path))
                shutil.move(image_path, error_root)
                shutil.move(json_path, error_root)
                continue

            # 获取保存图片路径和保存目录
            save_image_path = os.path.join(output_root, 'clips', os.path.relpath(image_path, images_root))
            save_folder_path = os.path.dirname(save_image_path)
            save_mask_path = os.path.join(save_folder_path, os.path.splitext(os.path.basename(save_image_path))[0] + '.png')
            
            if resize_size is not None:
                resize_width, resize_height = resize_size
                original_img = cv2.resize(original_img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            img_h,img_w,_ = original_img.shape

            # 记录交点数据
            lane_intersections = []
            
            if mode != "error_train":
                if not os.path.exists(json_path):
                    print(f"Warning: JSON file {json_path} not found, skipping...")
                    shutil.move(image_path, error_root)
                    continue

                # get json info
                json_info = get_json_info(json_path)
                if json_info is None:
                    print("Warning: json path: {} is error".format(json_path))
                    shutil.move(image_path, error_root)
                    shutil.move(json_path, error_root)
                    continue

                # 处理每条车道线
                for lane in json_info:
                    points = lane['points']
                    label = lane['label']
                    lane_label = lane['lane_label']

                    points = np.array(points, dtype=np.float32)
                    scale_factors = np.array([img_w / original_img_w, img_h / original_img_h])
                    points = (points * scale_factors).astype(np.int32)

                    # points是从下到上排序的（y值从大到小）
                    points = points[points[:, 1].argsort()[::-1]]

                    # 计算并绘制交点
                    intersections = []
                    for y in ROW_COORDS:
                        for i in range(len(points) - 1):
                            p1 = points[i]
                            p2 = points[i + 1]
                            point = find_intersection(p1, p2, y)
                            if point is not None:
                                intersections.append((point[0], point[1]))
                                break

                    lane_intersections.append({
                        'label': label,
                        'points': intersections,
                        'lane_label': lane_label,
                    })
                
                if not validate_info(lane_intersections):
                    print("Warning: info is invalid. Skipping {}".format(image_path))
                    shutil.move(image_path, error_root)
                    shutil.move(json_path, error_root)
                    continue

                # ---- 统计当前图片在每个y坐标上有几条车道线 ----
                total_images += 1
                for y in ROW_COORDS:
                    count_at_y = 0
                    for lane in lane_intersections:
                        for px, py in lane['points']:
                            if py == y and px >= 0:  # px >= 0 表示有效交点
                                count_at_y += 1
                                break
                    count_at_y = min(count_at_y, 4)  # 防止异常值
                    y_row_counts[y][count_at_y] += 1
                    y_row_total[y] += count_at_y
                    if count_at_y > 0:
                        y_row_img_count[y] += 1
            else:
                lane_intersections.append({
                        'label': -1,
                        'points': [[-2,-2]],
                        'lane_label': [],
                    })
            
            os.makedirs(save_folder_path, exist_ok=True)

            # 写label.json文件
            img_path = os.path.relpath(save_image_path, output_root)
            convert_to_json(img_path, lane_intersections, label_file)

            # 写gt.txt文件  保存掩码图片
            temp_cache_dict, bin_label, bin_lane_label_str = generate_segmentation_and_list2(img_path, lane_intersections, [img_w, img_h], save_mask_path)

            # 把点，线，序号画到图片上并保存
            draw_and_save_lanes(original_img, img_path, temp_cache_dict[img_path]['points'], ROW_COORDS, output_root)

            if mode == 'test':
                with open(gt_file, 'a') as f:
                    f.write(img_path + '\n')
            else:
                with open(gt_file, 'a') as f:
                    f.write(img_path + ' ' + os.path.relpath(save_mask_path, output_root) + ' ' + ' '.join(list(map(str, bin_label))) + bin_lane_label_str + '\n')

            cache_dict.update(temp_cache_dict) 

            # 保存图片
            cv2.imwrite(save_image_path, original_img)
    
    # ---- 打印统计结果 ----
    print(f"\n{'='*80}")
    print(f"[{mode}] 共处理 {total_images} 张图片, ROW_COORDS 共 {len(ROW_COORDS)} 个y坐标")
    print(f"{'='*80}")
    print(f"{'y坐标':>6} | {'0条线':>6} | {'1条线':>6} | {'2条线':>6} | {'3条线':>6} | {'4条线':>6} | {'有线图片数':>8} | {'平均线数':>8} | {'覆盖率':>8}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}-+-{'-'*12}-+-{'-'*10}")

    for y in ROW_COORDS:
        counts = y_row_counts[y]
        img_with_lane = y_row_img_count[y]
        avg_lanes = y_row_total[y] / total_images if total_images > 0 else 0
        coverage = img_with_lane / total_images * 100 if total_images > 0 else 0
        print(f"{y:>8} | {counts[0]:>8} | {counts[1]:>8} | {counts[2]:>8} | {counts[3]:>8} | {counts[4]:>8} | {img_with_lane:>13} | {avg_lanes:>12.2f} | {coverage:>10.1f}%")

    # 保存到文件
    stats_file = os.path.join(output_root, f'row_coords_stats_{mode}.txt')
    print(f"\n统计结果已保存到: {stats_file}")

    # 写tusimple_anno_cache.json文件
    with open(tusimple_anno_cache_file, 'w') as f:
        json.dump(cache_dict, f, indent=2, ensure_ascii=False)

def main(images_root, train_list, test_list, error_list, output_root, resize_size=None):
    '''
    '''
    dispose_data(images_root, train_list, output_root, resize_size, 'train')

    dispose_data(images_root, test_list, output_root, resize_size, 'test')

    dispose_data(images_root, error_list, output_root, resize_size, 'error_train')

if __name__ == "__main__":
    images_root = '../../data/model_use'
    train_list  = ["20260417", "20260421", "20260429", "20260520", "20260525", "20260527", "20260603", "20260630", "images"]
    test_list   = ["images_test"]
    error_list  = ['error_result']
    output_root = '../../data/model_use/TUSimple'
    resize_size = (640, 360)
    
    os.makedirs(output_root, exist_ok=True)
    
    main(images_root, train_list, test_list, error_list, output_root, resize_size)