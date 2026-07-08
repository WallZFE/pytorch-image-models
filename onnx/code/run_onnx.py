import os
import glob
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import base64
import json
import math
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROW_COORDS = [115, 118, 120, 123, 125, 128, 130, 133, 136, 139,
              141, 144, 147, 150, 153, 156, 160, 163, 166, 170,
              173, 177, 181, 184, 188, 192, 197, 201, 205, 209,
              214, 218, 223, 228, 233, 238, 243, 248, 253, 258,
              263, 269, 274, 280, 285, 291, 297, 302, 308, 314,
              320, 326, 333, 339, 345, 350, 355, 359]

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# 车道线属性标签
lane_label_index_dic = ["单", "多", "白", "黄", "实线", "虚线", "实虚线", "虚实线"]

def preprocess_image(img_path, train_height, train_width, crop_ratio):
    """
    读取并预处理单张图片
    :return: (img_numpy_CHW, original_h, original_w) 或 (None, 0, 0) 如果读取失败
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, 0, 0
    
    h, w, _ = img.shape
    resize_h = int(train_height / crop_ratio)
    resize_w = train_width
    
    img_resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    y_min = resize_h - train_height
    img_cropped = img_resized[y_min:resize_h, 0:resize_w]
    
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_float - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    
    return img_transposed, h, w

def collect_images(data_root, mode='video', test_txt_path=None):
    """
    收集所有待处理图片
    :param data_root: 图片数据的根目录 (对应 args.demo_data_root)
    :param mode: 运行模式 ('json', 'video', 'eval')
    :param test_txt_path: eval模式下的 test.txt 文件路径
    """
    image_paths = []

    if mode == 'eval':
        # ================= EVAL 模式：读取 test.txt =================
        if test_txt_path is None or not os.path.exists(test_txt_path):
            print(f"❌ 错误: eval 模式下必须提供有效的 test.txt 路径，当前为: {test_txt_path}")
            return []

        print(f"[EVAL模式] 正在从 {test_txt_path} 读取图片列表...")
        with open(test_txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                
                # 拼接完整路径并规范化 (处理不同操作系统的斜杠问题)
                full_path = os.path.normpath(os.path.join(data_root, rel_path))
                
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    # 如果找不到文件，打印警告（如果警告太多可以注释掉这行）
                    print(f"⚠️ 警告: 图片不存在，跳过 -> {full_path}")
                    
        print(f"✅ 共从 test.txt 成功加载 {len(image_paths)} 张有效图片")

    else:
        # ================= JSON/VIDEO 模式：Glob 扫描目录 =================
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        print(f"扫描目录 {data_root} 下的所有图片...")

        json_files_to_delete = []

        for file_path in glob.glob(os.path.join(data_root, "**", "*"), recursive=True):
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext in image_extensions:
                image_paths.append(file_path)
            elif mode == 'json' and ext == '.json':
                json_files_to_delete.append(file_path)

        print(f"✅ 共找到 {len(image_paths)} 张图片")

        # ================= JSON 模式专属：删除 JSON 文件 =================
        if mode == 'json':
            if json_files_to_delete:
                print(f"🗑️  [JSON模式] 发现 {len(json_files_to_delete)} 个 JSON 文件，正在删除...")
                
                deleted_count = 0
                failed_count = 0
                
                for json_path in tqdm(json_files_to_delete, desc="删除JSON文件", ncols=80):
                    try:
                        os.remove(json_path)
                        deleted_count += 1
                    except Exception as e:
                        failed_count += 1
                        print(f"\n⚠️ 删除失败: {json_path} - {e}")
                        
                print(f"✅ JSON 文件清理完成: 成功删除 {deleted_count} 个, 失败 {failed_count} 个")
            else:
                print("ℹ️  [JSON模式] 目录下未找到任何 JSON 文件")

    return image_paths

def softmax_np(x):
    """numpy实现softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_np_axis(x, axis):
    """沿指定轴做softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def log_softmax_np(x, axis):
    """沿指定轴做log_softmax"""
    max_x = np.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def sigmoid_np(x):
    """numpy实现sigmoid"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def one_hot_np(indices, num_classes):
    """numpy版one_hot编码"""
    result = np.zeros(indices.shape + (num_classes,), dtype=np.float32)
    valid = (indices >= 0) & (indices < num_classes)
    flat_indices = indices[valid].astype(np.int64)
    flat_result = result[valid]  # copy
    flat_result[np.arange(len(flat_indices)), flat_indices] = 1.0
    result[valid] = flat_result  # write back
    return result


def smooth_l1_loss_np(pred, target, beta=1.0):
    """numpy版 smooth_l1_loss (reduction='none')"""
    diff = np.abs(pred - target)
    loss = np.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss


# ======================== Loss 函数 ========================

def soft_nll_np(pred, target, ignore_index=-1):
    """
    numpy版 soft_nll
    pred:   (B, C, H, W)  log_softmax后的值
    target: (B, H, W)     int64 labels
    """
    B, C, H, W = pred.shape
    target = target.astype(np.int64)
    invalid_target = (target == ignore_index)

    ttarget = target.copy()
    ttarget[invalid_target] = C

    target_l = target.copy() - 1
    target_r = target.copy() + 1

    invalid_part_l = (target_l == -1)
    invalid_part_r = (target_r == C)

    target_l[invalid_target | invalid_part_l] = C
    target_r[invalid_target | invalid_part_r] = C

    supp_part_l = target.copy()
    supp_part_r = target.copy()
    supp_part_l[target != 0] = C
    supp_part_r[target != C - 1] = C

    def _to_onehot(t):
        oh = one_hot_np(t, C + 1)[..., :-1]  # 去掉额外类别
        return np.transpose(oh, (0, 3, 1, 2))  # (B, C, H, W)

    target_onehot = _to_onehot(ttarget)
    target_l_onehot = _to_onehot(target_l)
    target_r_onehot = _to_onehot(target_r)
    supp_part_l_onehot = _to_onehot(supp_part_l)
    supp_part_r_onehot = _to_onehot(supp_part_r)

    target_fusion = (0.9 * target_onehot
                     + 0.05 * target_l_onehot
                     + 0.05 * target_r_onehot
                     + 0.05 * supp_part_l_onehot
                     + 0.05 * supp_part_r_onehot)

    valid_count = np.sum(target != ignore_index)
    if valid_count == 0:
        return 0.0
    return float(-np.sum(target_fusion * pred) / valid_count)


def softmax_focal_loss_np(logits, labels, gamma=2, ignore_lb=-1):
    """
    numpy版 SoftmaxFocalLoss
    logits: (B, C, H, W)
    labels: (B, H, W) int64
    """
    scores = softmax_np_axis(logits, axis=1)
    factor = np.power(1.0 - scores, gamma)
    log_score = log_softmax_np(logits, axis=1)
    log_score = factor * log_score
    return soft_nll_np(log_score, labels, ignore_index=ignore_lb)


def cross_entropy_np(logits, labels):
    """
    numpy版 CrossEntropyLoss (reduction='mean')
    logits: (B, C, ...)  原始logits, 任意空间维度
    labels: (B, ...)     int64, 值域 [0, C-1]
    """
    log_probs = log_softmax_np(logits, axis=1)  # (B, C, ...)
    
    # 用 np.take_along_axis 替代手动构造索引 —— 更简洁且不出错
    labels_expanded = np.expand_dims(labels.astype(np.int64), axis=1)  # (B, 1, ...)
    selected = np.take_along_axis(log_probs, labels_expanded, axis=1)   # (B, 1, ...)
    selected = selected.squeeze(axis=1)  # (B, ...)

    return float(-np.mean(selected))


def parsing_relation_loss_np(logits):
    """
    numpy版 ParsingRelationLoss
    logits: (B, C, H, W)
    """
    n, c, h, w = logits.shape
    # 相邻行做差: logits[:,:,i,:] - logits[:,:,i+1,:]
    diff = logits[:, :, :-1, :] - logits[:, :, 1:, :]  # (B, C, H-1, W)
    diff_flat = diff.reshape(-1)  # 展平
    # smooth_l1_loss(diff_flat, zeros)
    loss = smooth_l1_loss_np(diff_flat, np.zeros_like(diff_flat))
    return float(np.mean(loss))


def parsing_relation_dis_np(x):
    """
    numpy版 ParsingRelationDis
    x: (B, dim, num_rows, num_cols)
    """
    n, dim, num_rows, num_cols = x.shape

    # softmax 去掉最后一个通道
    x_softmax = softmax_np_axis(x[:, :dim-1, :, :], axis=1)  # (B, dim-1, num_rows, num_cols)

    # embedding: [0, 1, 2, ..., dim-2]
    embedding = np.arange(dim - 1, dtype=np.float32).reshape(1, -1, 1, 1)  # (1, dim-1, 1, 1)

    # pos = sum(x * embedding, dim=1) -> (B, num_rows, num_cols)
    pos = np.sum(x_softmax * embedding, axis=1)

    # diff_list: pos[:, i, :] - pos[:, i+1, :] for i in range(num_rows // 2)
    half_rows = num_rows // 2
    diffs = pos[:, :half_rows, :] - pos[:, 1:half_rows+1, :]  # (B, half_rows, num_cols)

    # loss = mean of L1 between consecutive diffs
    loss = 0.0
    num_diffs = half_rows - 1  # = len(diff_list) - 1
    if num_diffs <= 0:
        return 0.0
    for i in range(num_diffs):
        loss += float(np.mean(np.abs(diffs[:, i, :] - diffs[:, i+1, :])))
    loss /= num_diffs
    return float(loss)


def mean_loss_np(logits, label):
    """
    numpy版 MeanLoss
    logits: (B, C, H, W)
    label:  (B, H, W) int64, -1表示无效
    """
    B, C, H, W = logits.shape
    grid = np.arange(C, dtype=np.float32).reshape(1, C, 1, 1)

    probs = softmax_np_axis(logits, axis=1)
    pred_pos = np.sum(probs * grid, axis=1)  # (B, H, W)

    label_float = label.astype(np.float32)
    loss_all = smooth_l1_loss_np(pred_pos, label_float)

    valid_mask = (label != -1)
    valid_losses = loss_all[valid_mask]

    if len(valid_losses) == 0:
        return 0.0
    return float(np.mean(valid_losses))


def bce_with_logits_loss_np(logits, targets):
    """numpy版 BCEWithLogitsLoss (reduction='mean')"""
    logits = logits.astype(np.float64)
    targets = targets.astype(np.float64)
    max_val = np.maximum(logits, 0)
    loss = max_val - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
    return float(np.mean(loss))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ONNX车道线检测推理')
    parser.add_argument('--mode', type=str, default='json', choices=['json', 'video', 'eval'], help='运行模式: json=保存坐标JSON, video=生成视频, eval=计算测试集Loss')
    parser.add_argument('--onnx_model', type=str, default="../model/output_model.onnx", help='ONNX模型文件路径')
    parser.add_argument('--demo_data_root', type=str, default="../../../data/dispose", help='图片目录路径')
    parser.add_argument('--test_txt', type=str, default="../../../data/model_use/TUSimple/test.txt", help='eval模式下使用的 test.txt 图片列表文件路径')
    parser.add_argument('--cache_path', type=str, default="../../../data/model_use/TUSimple/tusimple_anno_cache_test.json", help='测试集GT标注缓存JSON路径 (eval模式需要)')
    parser.add_argument('--train_height', type=int, default=288, help='模型输入高度')
    parser.add_argument('--train_width', type=int, default=640, help='模型输入宽度')
    parser.add_argument('--crop_ratio', type=float, default=0.8, help='裁剪比例')
    parser.add_argument('--batch_size', type=int, default=1, help='推理batch size')
    parser.add_argument('--save_path', type=str, default='../test_results/', help='结果保存路径')
    parser.add_argument('--fps', type=float, default=25.0, help='视频帧率')
    return parser.parse_args()

def load_onnx_model(model_path, use_gpu=True):
    """
    加载ONNX模型
    :return: onnxruntime session, 输入/输出名称信息
    """
    providers = ['CPUExecutionProvider']
    if use_gpu:
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            sess = ort.InferenceSession(model_path, providers=providers)
            active_providers = sess.get_providers()
            print(f"ONNX模型加载成功,使用设备: {active_providers}")
        except Exception as e:
            print(f"GPU加载失败, 回退到CPU: {e}")
            providers = ['CPUExecutionProvider']
            sess = ort.InferenceSession(model_path, providers=providers)
    else:
        sess = ort.InferenceSession(model_path, providers=providers)

    # 获取输入输出信息
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    output_names = [o.name for o in sess.get_outputs()]
    output_shapes = [o.shape for o in sess.get_outputs()]

    print(f"输入名称: {input_name}, 输入形状: {input_shape}")
    print(f"输出名称: {output_names}")
    print(f"输出形状: {output_shapes}")

    return sess, input_name, output_names


def onnx_inference(session, input_name, output_names, input_data):
    """
    ONNX推理
    :param input_data: numpy array, shape (B, C, H, W)
    :return: dict of numpy arrays
    """
    outputs = session.run(output_names, {input_name: input_data})
    # 将输出映射为字典
    # ONNX模型输出顺序需要和模型导出时一致
    # 通常顺序: loc_row, loc_col, exist_row, exist_col, lane_label
    # 如果你的模型输出顺序不同，请在这里修改
    pred = {}
    if len(output_names) == 3:
        for name, val in zip(output_names, outputs):
            pred[name] = val
    else:
        # 默认假设顺序
        default_keys = ['loc_row', 'loc_col', 'exist_row', 'exist_col', 'lane_label']
        for key, val in zip(default_keys[:len(outputs)], outputs):
            pred[key] = val
    return pred


def find_output_key(pred, target_key):
    """
    从pred字典中查找包含target_key的输出
    支持输出名称带前缀的情况，如 'output_loc_row' 匹配 'loc_row'
    """
    # 先精确匹配
    if target_key in pred:
        return pred[target_key]
    # 模糊匹配
    for k, v in pred.items():
        if target_key in k:
            return v
    raise KeyError(f"找不到输出键: {target_key}, 可用键: {list(pred.keys())}")


# ==================== 核心推理后处理 ====================

def pred2coords(pred, local_width=5, original_image_widths=None, original_image_heights=None, train_width=None, train_height=None, resize_mode='original'):
    """
    预测结果转坐标
    :param pred: ONNX输出的字典
    :param original_image_widths/heights: 每个样本的原始图片尺寸（列表）
    :param train_width/train_height: 模型输入尺寸
    :param resize_mode: 对坐标映射到原始图片尺寸或模型输入尺寸，默认原始图片尺寸
    :return: all_coords, all_lane_labels
    """
    loc_row = find_output_key(pred, 'loc_row')
    exist_row = find_output_key(pred, 'exist_row')

    batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape

    max_indices_row = np.argmax(loc_row, axis=1)  # (B, num_cls_row, num_lane_row)
    valid_row = np.argmax(exist_row, axis=1)       # (B, num_cls_row, num_lane_row)

    # 获取车道线属性标签
    lane_label_raw = find_output_key(pred, 'lane_label')
    lane_label_prob = sigmoid_np(lane_label_raw)  # (B, 4, 8)
    lane_label = (lane_label_prob > 0.5).astype(np.uint8)

    # 预分配结果
    all_coords = [[] for _ in range(batch_size)]
    all_lane_labels = [{} for _ in range(batch_size)]

    row_lane_idx = [0, 1, 2, 3]

    for b in range(batch_size):
        img_w = original_image_widths[b]
        img_h = original_image_heights[b]

        coords = []
        lane_labels = {}

        for i in row_lane_idx:
            tmp = []
            if valid_row[b, :, i].sum() > 0:
                for k in range(valid_row.shape[1]):
                    if valid_row[b, k, i]:
                        idx_center = int(max_indices_row[b, k, i])
                        all_ind = list(range(max(0, idx_center - local_width),
                                             min(num_grid_row - 1, idx_center + local_width) + 1))
                        all_ind = np.array(all_ind, dtype=np.int64)

                        raw_vals = loc_row[b, all_ind, k, i]  # (local_width*2+1,)
                        probs = softmax_np(raw_vals)
                        out_tmp = np.sum(probs * all_ind.astype(np.float32))
                        if resize_mode == "original":
                            out_tmp = out_tmp / (num_grid_row - 1) * img_w
                            y_val = int(ROW_COORDS[k] / 360.0 * img_h)
                        else:
                            out_tmp = out_tmp / (num_grid_row - 1) * train_width
                            y_val = int(ROW_COORDS[k])
                        tmp.append((int(out_tmp), y_val))

            coords.append(tmp)

            for tmp_n in range(len(lane_label_index_dic)):
                if lane_label[b, i, tmp_n]:
                    if str(i) not in lane_labels:
                        lane_labels[str(i)] = []
                    lane_labels[str(i)].append(lane_label_index_dic[tmp_n])

        all_coords[b] = coords
        all_lane_labels[b] = lane_labels

    return all_coords, all_lane_labels


# ==================== JSON 模式 ====================

def douglas_peucker(points, epsilon):
    """
    Douglas-Peucker 折线简化算法
    points: [(x, y), ...] 有序点列
    epsilon: 容差，越大点越少
    """
    if len(points) <= 2:
        return points

    # 找到离首尾连线最远的点
    start, end = points[0], points[-1]
    max_dist = 0
    max_idx = 0

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    line_len_sq = dx * dx + dy * dy

    for i in range(1, len(points) - 1):
        px, py = points[i]
        if line_len_sq == 0:
            dist = ((px - start[0])**2 + (py - start[1])**2) ** 0.5
        else:
            t = max(0, min(1, ((px - start[0]) * dx + (py - start[1]) * dy) / line_len_sq))
            proj_x = start[0] + t * dx
            proj_y = start[1] + t * dy
            dist = ((px - proj_x)**2 + (py - proj_y)**2) ** 0.5

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # 如果最大距离超过容差，递归简化
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def process_single_image_json(img_path, coords, lane_labels, img_h, img_w, need_base64=True):
    """处理单个图像并生成 labelme 风格 JSON"""
    img = cv2.imread(img_path)

    json_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(img_path),
        "imageHeight": img_h,
        "imageWidth": img_w
    }

    if need_base64 and img is not None:
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        json_data["imageData"] = base64.b64encode(buffer).decode('utf-8')

    for num, index in enumerate([0, 1, 2, 3]):
        lane = coords[num]
        if len(lane) == 0:
            continue

        lane_list = [list(pt) for pt in lane]
        lane_sorted = sorted(lane_list, key=lambda pt: pt[1])

        if len(lane_sorted) > 2:
            final_points = douglas_peucker(lane_sorted, epsilon=3.5)
        else:
            final_points = lane_sorted

        tmp_shapes = {}
        if str(index) not in lane_labels:
            tmp_shapes['label'] = ' '
        else:
            tmp_shapes['label'] = ' '.join(lane_labels[str(index)])

        tmp_shapes['points'] = final_points
        tmp_shapes['group_id'] = index + 1
        tmp_shapes['description'] = ''
        tmp_shapes['shape_type'] = 'linestrip'
        tmp_shapes['flags'] = {}
        tmp_shapes["mask"] = None

        json_data['shapes'].append(tmp_shapes)

    return json_data


def run_json_mode(session, input_name, output_names, image_list, args):
    """JSON模式: 对每张图片生成坐标JSON文件"""
    batch_size = args.batch_size

    total_batches = math.ceil(len(image_list) / batch_size)
    print(f'[JSON模式] 共 {len(image_list)} 张图片, {total_batches} 个batch')

    for batch_start in tqdm(range(0, len(image_list), batch_size), total=total_batches, desc='处理中'):
        batch_paths = image_list[batch_start:batch_start + batch_size]

        # 预处理batch
        batch_numpys = []
        original_widths = []
        original_heights = []
        valid_paths = []

        for img_path in batch_paths:
            img_transposed, h, w = preprocess_image(img_path, args.train_height, args.train_width, args.crop_ratio)

            original_heights.append(h)
            original_widths.append(w)
            valid_paths.append(img_path)
            batch_numpys.append(img_transposed)

        if len(batch_numpys) == 0:
            continue

        # 组batch并推理
        input_data = np.stack(batch_numpys, axis=0).astype(np.float32)
        pred = onnx_inference(session, input_name, output_names, input_data)

        # 后处理
        all_coords, all_lane_labels = pred2coords(pred, original_image_widths=original_widths,original_image_heights=original_heights)

        # 保存JSON
        for b in range(len(valid_paths)):
            img_path = valid_paths[b]
            img_h = original_heights[b]
            img_w = original_widths[b]

            if img_h == 0 or img_w == 0:
                continue

            json_data = process_single_image_json(img_path, all_coords[b], all_lane_labels[b], img_h, img_w)

            output_json_path = os.path.join(os.path.dirname(img_path),os.path.splitext(os.path.basename(img_path))[0] + '.json')

            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error saving JSON for {img_path}: {e}")

    print("✅ JSON模式处理完成!")


# ==================== 视频模式 ====================

def draw_lane_on_image(vis, coords, filter_coords, lane_labels):
    """
    在图像上绘制车道线和标签
    """

    for num, lane in enumerate(coords):
        if len(lane) == 0:
            continue
        for coord in lane:
            if num == 0:
                cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (255, 0, 0), -1)
            elif num == 1:
                cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 140, 255), -1)
            elif num == 2:
                cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 255, 255), -1)
            else:
                cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 255, 128), -1)
    
    for lane in filter_coords:
        if len(lane) == 0:
            continue
        for num, coord in enumerate(lane):
            if num == 0:
                continue
            cv2.line(vis, (int(lane[num - 1][0]), int(lane[num - 1][1])) , (int(lane[num][0]), int(lane[num][1])),  (0, 255, 0), 7)

    # 绘制标签文字（使用PIL支持中文）
    sorted_keys = sorted(lane_labels.keys(), key=lambda x: int(x))
    for tmp_i, lane_index in enumerate(sorted_keys):
        label = lane_labels[lane_index]
        draw_point = (20, 40 + tmp_i * 80)

        vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_pil)
        # 尝试加载中文字体，失败则使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 70)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc", 70)
            except:
                try:
                    font = ImageFont.truetype("simhei.ttf", 70)
                except:
                    font = ImageFont.load_default()
        text = f"车道线 {lane_index} 类型: {label}"
        draw.text(draw_point, text, font=font, fill=(255, 0, 0))
        vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)

    return vis

def lane_post_optimize(all_coords, original_widths, original_heights, input_width, input_height):
    '''
    '''
    # 去除异常点

    # 拟合
    
    # 映射到原图中
    for b in range(len(all_coords)):
        scale_x = original_widths[b] / input_width
        scale_y = original_heights[b] / input_height

        for i in range(len(all_coords[b])):
            lane = all_coords[b][i]
            if len(lane) == 0:
                continue
            pts = np.array(lane, dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            all_coords[b][i] = pts

    all_lane_labels = all_coords
    return all_coords, all_lane_labels


def run_video_mode(session, input_name, output_names, image_list, args):
    """视频模式：将带车道线的图片合成视频"""
    batch_size = args.batch_size 
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    image_list = natsorted(image_list, key=os.path.basename)

    # 先读取第一张图获取尺寸
    first_img = cv2.imread(image_list[0])
    if first_img is None:
        print(f"Error: 无法读取图片 {image_list[0]}")
        return
    img_h, img_w, _ = first_img.shape

    save_name = os.path.basename(args.demo_data_root.rstrip('/'))
    video_path = os.path.join(save_path, f'{save_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter(video_path, fourcc, args.fps, (img_w, img_h))
    print(f'[视频模式] 共 {len(image_list)} 帧, 保存到 {video_path}')

    for batch_start in tqdm(range(0, len(image_list), batch_size), total=math.ceil(len(image_list) / batch_size), desc='生成视频'):
        batch_paths = image_list[batch_start:batch_start + batch_size]

        batch_numpys = []
        original_widths = []
        original_heights = []
        valid_paths = []

        for img_path in batch_paths:
            img_transposed, h, w = preprocess_image(img_path, args.train_height, args.train_width, args.crop_ratio)

            original_heights.append(h)
            original_widths.append(w)
            valid_paths.append(img_path)
            batch_numpys.append(img_transposed)

        if len(batch_numpys) == 0:
            continue

        input_data = np.stack(batch_numpys, axis=0).astype(np.float32)
        pred = onnx_inference(session, input_name, output_names, input_data)

        all_coords, all_lane_labels = pred2coords(pred, original_image_widths=original_widths, original_image_heights=original_heights, train_width=args.train_width, resize_mode='train')

        all_coords, all_filter_coords = lane_post_optimize(all_coords, original_widths, original_heights, args.train_width, args.train_height/args.crop_ratio)
       
        for b in range(len(valid_paths)):
            img_path = valid_paths[b]
            vis = cv2.imread(img_path)
            if vis is None:
                continue

            vis = draw_lane_on_image(vis, all_coords[b], all_filter_coords[b], all_lane_labels[b])
            vout.write(vis)

    vout.release()
    print("✅ 视频模式处理完成!")

# ==================== 获取真实信息 ====================

def my_interp_cpu(points, interp_loc, direction=0):
    """
    numpy版插值
    :param points: (lane_num, point_num, 2)
    :param interp_loc: (new_point_num,)
    :param direction: 0=按y插值x, 1=按x插值y
    :return: (lane_num, new_point_num, 2)
    """
    lane_num, point_num, _ = points.shape
    new_point_num = len(interp_loc)
    output = np.full((lane_num, new_point_num, 2), -1.0, dtype=np.float32)

    inv_dir = 1 - direction

    for l in range(lane_num):
        lane = points[l]
        lane_dir = lane[:, direction]
        lane_inv = lane[:, inv_dir]

        for k, current_loc in enumerate(interp_loc):
            output[l, k, inv_dir] = current_loc
            pos = -1

            for i in range(point_num - 1, 0, -1):
                v1 = lane_inv[i]
                v2 = lane_inv[i - 1]
                if lane_dir[i] < 0 or lane_dir[i - 1] < 0 or v1 < 0 or v2 < 0:
                    continue
                if (v1 - current_loc) * (v2 - current_loc) <= 0:
                    pos = i
                    break

            if pos == -1:
                continue

            p1_dir, p0_dir = lane_dir[pos], lane_dir[pos - 1]
            p1_inv, p0_inv = lane_inv[pos], lane_inv[pos - 1]

            length = abs(p1_inv - p0_inv)
            if length < 1e-6:
                continue

            factor1 = 1.0 - abs(p1_inv - current_loc) / length
            factor2 = 1.0 - factor1
            output[l, k, direction] = p1_dir * factor1 + p0_dir * factor2

    return output

def load_gt_labels(image_list, data_root, cache_path, num_cell_row=100, row_lane_idx=[0, 1, 2, 3]):
    """
    从GT缓存文件加载ground truth标签

    :return: dict of image_name -> target dict
    """
    with open(cache_path, 'r') as f:
        cached_points = json.load(f)

    labels_dict = {}

    for img_path in tqdm(image_list, desc="加载数据"):
        img_name = os.path.relpath(img_path, data_root)

        if img_name not in cached_points:
            img_name_try = os.path.basename(img_path)
            if img_name_try in cached_points:
                img_name = img_name_try
            else:
                print(f"Warning: {img_path} 的GT不存在于cache中,跳过")
                continue

        infos = cached_points[img_name]
        points = np.array(infos["points"]).astype(np.float32)  # (num_lanes, num_points, 2)
        lane_label = np.array(infos["lane_label"]).astype(np.float32)  # (num_lanes, num_attrs)

        # 读取图片获取原始尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape

        target = {
            "labels_row": None,
            "labels_row_float": None,
            "row_coords": None,
            "lane_label": lane_label,
            "img_w": img_w,
            "img_h": img_h,
        }

        # row 行
        # points_row shape: (num_lanes, H_row, 2)
        points_row = my_interp_cpu(points, ROW_COORDS, direction=0)

        # points_row_extend shape: (H_row, num_lanes) —— 取 x 坐标并转置
        points_row_extend = points_row[:, :, 0].T  # numpy transpose

        # labels_row: 离散化为整数标签
        labels_row = (points_row_extend / img_w * (num_cell_row - 1)).astype(np.int64)
        labels_row[(points_row_extend < 0) | (points_row_extend > img_w)] = -1
        labels_row[(labels_row < 0) | (labels_row > (num_cell_row - 1))] = -1

        # labels_row_float: 归一化浮点标签
        labels_row_float = points_row_extend / img_w
        labels_row_float[(labels_row_float < 0) | (labels_row_float > 1)] = -1

        # 填充 target
        target["labels_row"] = labels_row
        target["labels_row_float"] = labels_row_float

        # Row Coords (选取指定 lane_idx，默认 [1, 2])
        pts_row = points[row_lane_idx, :, :]   # (len(row_lane_idx), num_points, 2)
        interp_row = my_interp_cpu(pts_row, ROW_COORDS, direction=0)  # (len(row_lane_idx), H_row, 2)
        row_coords = interp_row.astype(np.float32)

        invalid_x = (row_coords[..., 0] < 0) | (row_coords[..., 0] > img_w)
        row_coords[..., 0] = np.where(invalid_x, -10000.0, row_coords[..., 0])

        target["row_coords"] = row_coords

        labels_dict[img_path] = target

    return labels_dict

# ==================== 主Loss函数 ====================

def compute_test_loss(pred, target):
    """
    计算测试阶段的loss

    :param pred: dict, ONNX模型输出字典
        - 'loc_row':     (B, C_row, H_row, num_lane_row)
        - 'exist_row':   (B, 2, H_row, num_lane_row)
        - 'lane_label':  (B, num_lanes, num_attrs)   ← 原始logits

    :param target: dict, ground truth标签字典
        - 'labels_row':    (B, H_row, num_lane_row)  int64, 值域 [-1, C_row-1]
        - 'lane_label':    (B, num_lanes, num_attrs) float32, 0/1标签

    :return: (total_loss, loss_items_dict)
    """
    # ---- 权重配置 ----
    cls_loss_weight = 1.0
    relation_loss_weight = 0.0
    relation_dis_weight = 0.0
    cls_ext_weight = 1.0
    mean_loss_row_weight = 1.0
    lane_attr_loss_weight = 1.0

    # exist labels: 0/1 整数
    cls_out_ext_label = (target['labels_row'] != -1).astype(np.int64)    # (1, H_row, num_lanes)

    total_loss = 0.0
    loss_items = {}

    # 1. cls_loss (row)
    if cls_loss_weight != 0:
        loss_cur = softmax_focal_loss_np(pred['loc_row'], target['labels_row'])
        loss_items['cls_loss'] = loss_cur
        total_loss += loss_cur * cls_loss_weight

    # 2. relation_loss
    if relation_loss_weight != 0:
        loss_cur = parsing_relation_loss_np(pred['loc_row'])
        loss_items['relation_loss'] = loss_cur
        total_loss += loss_cur * relation_loss_weight

    # 3. relation_dis
    if relation_dis_weight != 0:
        loss_cur = parsing_relation_dis_np(pred['loc_row'])
        loss_items['relation_dis'] = loss_cur
        total_loss += loss_cur * relation_dis_weight

    # 4. cls_ext (exist_row)
    if cls_ext_weight != 0:
        loss_cur = cross_entropy_np(pred['exist_row'], cls_out_ext_label)
        loss_items['cls_ext'] = loss_cur
        total_loss += loss_cur * cls_ext_weight

    # 5. mean_loss_row
    if mean_loss_row_weight != 0:
        loss_cur = mean_loss_np(pred['loc_row'], target['labels_row'])
        loss_items['mean_loss_row'] = loss_cur
        total_loss += loss_cur * mean_loss_row_weight

    # 6. lane_attr_loss (BCE)
    if lane_attr_loss_weight != 0:
        loss_cur = bce_with_logits_loss_np(pred['lane_label'], target['lane_label'])
        loss_items['lane_attr_loss'] = loss_cur
        total_loss += loss_cur * lane_attr_loss_weight

    loss_items['total_loss'] = total_loss
    return total_loss, loss_items

def process_row_lanes(loc_row, valid_row, max_idx_row, lane_indices, row_anchor_t, img_w, num_grid_row, H_row, local_width):
    B = loc_row.shape[0]
    num_lanes = len(lane_indices)
    lane_idx_arr = np.array(lane_indices, dtype=np.int64)

    v = valid_row[:, :, lane_idx_arr]
    m = max_idx_row[:, :, lane_idx_arr]

    if v.shape[1] >= H_row:
        v = v[:, :H_row, :]
        m = m[:, :H_row, :]
    else:
        pad_h = H_row - v.shape[1]
        v = np.pad(v, ((0, 0), (0, pad_h), (0, 0)), constant_values=False)
        m = np.pad(m, ((0, 0), (0, pad_h), (0, 0)), constant_values=0)

    v = v.astype(bool)
    m = m.astype(np.int64)

    offsets = np.arange(-local_width, local_width + 1)
    all_ind = m[..., np.newaxis] + offsets.reshape(1, 1, 1, -1)
    all_ind = np.clip(all_ind, 0, num_grid_row - 1)

    b_idx = np.arange(B).reshape(B, 1, 1, 1)
    h_idx = np.arange(H_row).reshape(1, H_row, 1, 1)
    l_idx = lane_idx_arr.reshape(1, 1, num_lanes, 1)

    b_e = np.broadcast_to(b_idx, (B, H_row, num_lanes, all_ind.shape[-1]))
    h_e = np.broadcast_to(h_idx, (B, H_row, num_lanes, all_ind.shape[-1]))
    l_e = np.broadcast_to(l_idx, (B, H_row, num_lanes, all_ind.shape[-1]))

    logits = loc_row[b_e, all_ind, h_e, l_e]
    weights = softmax_np_axis(logits, axis=-1)
    refined = np.sum(weights * all_ind.astype(np.float32), axis=-1) + 0.5

    x = refined / (num_grid_row - 1) * img_w
    x = x.transpose(0, 2, 1)
    y = np.broadcast_to(row_anchor_t.reshape(1, 1, H_row), (B, num_lanes, H_row))

    coords = np.stack([x, y], axis=-1)

    invalid = ~v.transpose(0, 2, 1)
    coords[..., 1] = np.where(invalid, y.astype(np.float32), coords[..., 1])
    coords[..., 0] = np.where(invalid, np.full_like(coords[..., 0], -10000.0), coords[..., 0])

    return coords

def pred2coords_row_col(pred, row_anchor, col_anchor, image_widths, image_heights, local_width=1):
    B = pred['loc_row'].shape[0]
    num_grid_row = pred['loc_row'].shape[1]
    H_row = len(row_anchor)

    row_anchor_t = np.array(row_anchor, dtype=np.float32)

    if isinstance(image_widths, np.ndarray):
        img_w = image_widths.astype(np.float32).reshape(-1)
        if img_w.size == 1:
            img_w = np.broadcast_to(img_w, (B,))
        img_w = img_w.reshape(B, 1, 1)
    elif isinstance(image_widths, (int, float)):
        img_w = np.full((B, 1, 1), float(image_widths), dtype=np.float32)
    else:
        # list / tuple
        img_w = np.array(image_widths, dtype=np.float32).reshape(-1)
        if img_w.size == 1:
            img_w = np.broadcast_to(img_w, (B,))
        img_w = img_w.reshape(B, 1, 1)

    if isinstance(image_heights, np.ndarray):
        img_h = image_heights.astype(np.float32).reshape(-1)
        if img_h.size == 1:
            img_h = np.broadcast_to(img_h, (B,))
        img_h = img_h.reshape(B, 1, 1)
    elif isinstance(image_heights, (int, float)):
        img_h = np.full((B, 1, 1), float(image_heights), dtype=np.float32)
    else:
        img_h = np.array(image_heights, dtype=np.float32).reshape(-1)
        if img_h.size == 1:
            img_h = np.broadcast_to(img_h, (B,))
        img_h = img_h.reshape(B, 1, 1)

    max_idx_row = np.argmax(pred['loc_row'], axis=1)

    valid_row = np.argmax(pred['exist_row'], axis=1)

    lane_label = None
    if 'lane_label' in pred and pred['lane_label'] is not None:
        lane_label = sigmoid_np(pred['lane_label']) > 0.5

    row_lane_idx = [0, 1, 2, 3]
    row_coords = process_row_lanes(pred['loc_row'], valid_row, max_idx_row, row_lane_idx, row_anchor_t, img_w, num_grid_row, H_row, local_width)
    
    col_coords = []
    
    return row_coords, col_coords, lane_label


def _calc_f1(tp, fp, fn):
    pr = tp / max(tp + fp, 1e-6)
    re = tp / max(tp + fn, 1e-6)
    f1 = 2 * pr * re / max(pr + re, 1e-6)
    return pr, re, f1

def lane_compute_metrics(c):
    """从累积的原始计数计算所有指标"""
    # Lane label
    ll_lane_acc = c['ll_lane_correct'] / max(c['ll_lane_total'], 1e-6)

    total_attr = c['ll_attr_tp'] + c['ll_attr_fp'] + c['ll_attr_fn'] + c['ll_attr_tn']
    ll_attr_acc = (c['ll_attr_tp'] + c['ll_attr_tn']) / max(total_attr, 1e-6)
    ll_attr_pr  = c['ll_attr_tp'] / max(c['ll_attr_tp'] + c['ll_attr_fp'], 1e-6)
    ll_attr_re  = c['ll_attr_tp'] / max(c['ll_attr_tp'] + c['ll_attr_fn'], 1e-6)
    ll_attr_f1  = 2 * ll_attr_pr * ll_attr_re / max(ll_attr_pr + ll_attr_re, 1e-6)

    # Row
    row_pr, row_re, row_f1 = _calc_f1(c['row_tp'], c['row_fp'], c['row_fn'])

    # Total
    lane_total_f1 = 0.3 * ll_attr_f1 + 0.7 * row_f1

    return {
        'll_lane_acc':       ll_lane_acc,
        'll_attr_acc':       ll_attr_acc,
        'll_attr_precision': ll_attr_pr,
        'll_attr_recall':    ll_attr_re,
        'll_attr_f1':        ll_attr_f1,
        'row_precision':     row_pr,
        'row_recall':        row_re,
        'row_f1':            row_f1,
        'lane_total_f1':     lane_total_f1,
    }

def lane_test(pred, gt, train_width, train_height):
    """
    全 NumPy 向量化评估。无 for 循环。
    """
    # 1. 获取预测结果
    row_coords, col_coords, lane_label = pred2coords_row_col(pred, ROW_COORDS, COL_COORDS, train_width, train_height)
        
    # 2. 获取 GT (确保是 numpy array)
    gt_row_coords = gt["row_coords"] if isinstance(gt["row_coords"], np.ndarray) else np.array(gt["row_coords"])
    gt_lane_label = gt["lane_label"] if isinstance(gt["lane_label"], np.ndarray) else np.array(gt["lane_label"])

    B = row_coords.shape[0]

    # ================= 1. Lane Label 评估 =================
    p_lab = lane_label.astype(bool)
    g_lab = gt_lane_label.astype(bool)
    
    # 指标 A: 车道线级准确率
    lane_match = np.all(p_lab == g_lab, axis=-1)
    ll_lane_correct = float(np.sum(lane_match))
    ll_lane_total = float(B * 4)
    
    # 指标 B: 属性级准确率
    ll_attr_tp = float(np.sum(p_lab & g_lab))
    ll_attr_fp = float(np.sum(p_lab & ~g_lab))
    ll_attr_fn = float(np.sum(~p_lab & g_lab))
    ll_attr_tn = float(np.sum(~p_lab & ~g_lab))

    # ================= 2. Row Coords 评估 =================
    p_row_x = row_coords[..., 0]
    g_row_x = gt_row_coords[..., 0]
    
    p_inv_r = (p_row_x <= -2.0)
    g_inv_r = (g_row_x <= -2.0)
    
    both_valid_r = (~p_inv_r) & (~g_inv_r)
    diff_ok_r = np.abs(p_row_x - g_row_x) <= math.ceil(train_width * 0.01)
    
    row_tp = float(np.sum(both_valid_r & diff_ok_r))
    row_fp = float(np.sum((~p_inv_r) & g_inv_r))
    row_fn = float(np.sum(p_inv_r & (~g_inv_r)))
    
    both_valid_wrong_r = both_valid_r & (~diff_ok_r)
    row_fp += float(np.sum(both_valid_wrong_r))
    row_fn += float(np.sum(both_valid_wrong_r))

    # ================= 只返回原始计数 =================
    results = {
        'll_lane_correct': ll_lane_correct,
        'll_lane_total':   ll_lane_total,

        'll_attr_tp': ll_attr_tp,
        'll_attr_fp': ll_attr_fp,
        'll_attr_fn': ll_attr_fn,
        'll_attr_tn': ll_attr_tn,

        'row_tp': row_tp,
        'row_fp': row_fp,
        'row_fn': row_fn,
    }
    return results

def run_eval_mode(session, input_name, output_names, image_list, args):
    """评估模式：计算测试集的 Loss 和 车道线指标"""
    print("\n" + "="*60)
    print("🚀 进入 EVAL 模式：计算测试集 Loss 与 车道线指标")
    print("="*60)
    
    # 动态获取模型输出的 grid 数量 (Channel数)，防止硬编码导致越界
    dummy_input = np.zeros((1, 3, args.train_height, args.train_width), dtype=np.float32)
    dummy_pred = onnx_inference(session, input_name, output_names, dummy_input)
    loc_row_shape = find_output_key(dummy_pred, 'loc_row').shape
    
    num_cell_row = loc_row_shape[1]  
    print(f"检测到模型 Grid 数量 -> Row: {num_cell_row}")

    # 加载 GT 标签
    print("正在加载 Ground Truth 标签并进行插值...")
    gt_labels = load_gt_labels(image_list, args.demo_data_root, args.cache_path, num_cell_row)
    
    valid_image_list = [p for p in image_list if p in gt_labels]
    if len(valid_image_list) == 0:
        print("❌ 错误：没有找到任何匹配的 GT 标签，请检查 --cache_path 和图片路径！")
        return

    print(f"成功匹配 {len(valid_image_list)} 张图片的 GT 标签。\n")

    # 开始推理并计算指标
    batch_size = args.batch_size
    total_batches = math.ceil(len(valid_image_list) / batch_size)
    
    # 1. 初始化 Loss 累加器
    all_losses = {
        'total_loss': 0.0, 'cls_loss': 0.0, 'relation_loss': 0.0, 'relation_dis': 0.0, 
        'cls_ext': 0.0,  'mean_loss_row': 0.0, 'lane_attr_loss': 0.0
    }
    
    # 2. 初始化 Lane 指标累加器 (对应 lane_test 返回的 keys)
    counts = {
        'll_lane_correct': 0.0, 'll_lane_total': 0.0,
        'll_attr_tp': 0.0, 'll_attr_fp': 0.0, 'll_attr_fn': 0.0, 'll_attr_tn': 0.0,
        'row_tp': 0.0, 'row_fp': 0.0, 'row_fn': 0.0,
    }
    
    valid_count = 0
    total_num = len(valid_image_list)

    for batch_start in tqdm(range(0, total_num, batch_size), total=total_batches, desc='评估中'):
        raw_batch_paths = valid_image_list[batch_start:batch_start + batch_size]

        batch_numpys, batch_paths, original_widths, original_heights = [], [], [], []

        for img_path in raw_batch_paths:
            img_transposed, h, w = preprocess_image(img_path, args.train_height, args.train_width, args.crop_ratio)

            original_heights.append(h)
            original_widths.append(w)
            batch_paths.append(img_path)
            batch_numpys.append(img_transposed)

        if len(batch_numpys) == 0: 
            continue

        # 推理
        input_data = np.stack(batch_numpys, axis=0).astype(np.float32)
        pred = onnx_inference(session, input_name, output_names, input_data)

        # 组装 Target (使用过滤后的 batch_paths)
        target = {
            'labels_row': np.stack([gt_labels[p]['labels_row'] for p in batch_paths], axis=0),
            'lane_label': np.stack([gt_labels[p]['lane_label'] for p in batch_paths], axis=0),
            'row_coords': np.stack([gt_labels[p]['row_coords'] for p in batch_paths], axis=0),
        }

        # 计算 Loss
        #! loss的统计方式不对 需要更换
        _, loss_items = compute_test_loss(pred, target)

        # 计算 Lane 指标
        lane_res_items = lane_test(pred, target, args.train_width, int(args.train_height / args.crop_ratio))

        # 累加 Loss
        for key in all_losses:
            if key in loss_items:
                all_losses[key] += loss_items[key]
                
        # 累加 Lane 指标
        for k in counts:
            counts[k] += lane_res_items[k]
                
        valid_count += 1

    # ================= 4. 打印最终结果 =================
    if valid_count > 0:
        print("\n" + "="*60)
        print(f"📊 评估完成 (共 {valid_count} 个 Batch, {total_num} 张图片)")
        
        # 打印 Loss
        print(f"\n🔹 【Loss 统计】")
        print(f"{'Loss 项':<20} {'平均值':>10}")
        print("-" * 35)
        for key, val in all_losses.items():
            avg = val / valid_count
            print(f"{key:<20} {avg:>10.4f}")
            
        # 打印 Lane 指标
        print(f"\n🔹 【Lane 指标统计】")
        print(f"{'指标项':<20} {'平均值':>10}")
        print("-" * 35)
        metrics = lane_compute_metrics(counts)
        for key, val in metrics.items():
            print(f"{key:<20} {val * 100:>9.6f}%")
            
        # 打印核心 F1
        core_f1 = metrics['lane_total_f1']
        print("\n" + "="*60)
        print(f"🏆 核心指标 Lane Total F1: {core_f1:.8f}")
        print("="*60 + "\n")
    else:
        print("❌ 没有有效数据计算 loss 和指标!")

# ==================== 主入口 ====================

def main():
    args = parse_args()

    # ---- 加载ONNX模型 ----
    session, input_name, output_names = load_onnx_model(args.onnx_model)

    # ---- 收集图片 ----
    image_list = collect_images(data_root=args.demo_data_root, mode=args.mode, test_txt_path=args.test_txt)
    if len(image_list) == 0:
        print("未找到任何图片，退出。")
        return

    # ---- 根据模式运行 ----
    if args.mode == 'json':
        run_json_mode(session, input_name, output_names, image_list, args)
    elif args.mode == 'video':
        run_video_mode(session, input_name, output_names, image_list, args)
    elif args.mode == 'eval':
        run_eval_mode(session, input_name, output_names, image_list, args)
    else:
        print(f"未知的模式: {args.mode}")

if __name__ == "__main__":
    main()