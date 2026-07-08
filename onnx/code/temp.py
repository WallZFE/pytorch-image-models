import os
import glob
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import math
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import medfilt
from sklearn.neighbors import LocalOutlierFactor
ROW_COORDS = [115, 118, 120, 123, 125, 128, 130, 133, 136, 139,
              141, 144, 147, 150, 153, 156, 160, 163, 166, 170,
              173, 177, 181, 184, 188, 192, 197, 201, 205, 209,
              214, 218, 223, 228, 233, 238, 243, 248, 253, 258,
              263, 269, 274, 280, 285, 291, 297, 302, 308, 314,
              320, 326, 333, 339, 345, 350, 355, 359]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
lane_label_index_dic = ["单", "多", "白", "黄", "实线", "虚线", "实虚线", "虚实线"]

class Kalman1D:
    """
    一维 Kalman Filter

    状态:
        x  : 当前x坐标
        vx : 横向速度

    说明：
        状态向量:
            X = [x, vx]^T

        每个row对应一个Kalman
        一个Lane共58个Kalman
        共4条Lane。
    """

    def __init__(
            self,
            process_noise_x=1.0,
            process_noise_v=0.5,
            measure_noise=4.0):

        self.initialized = False

        # 状态
        self.x = np.zeros((2, 1), dtype=np.float32)

        # 协方差
        self.P = np.eye(2, dtype=np.float32)

        # 状态转移矩阵
        self.F = np.array(
            [
                [1.0, 1.0],
                [0.0, 1.0]
            ],
            dtype=np.float32
        )

        # 观测矩阵
        self.H = np.array(
            [
                [1.0, 0.0]
            ],
            dtype=np.float32
        )

        # 保存基础Q
        self.Q_base = np.array(
            [
                [process_noise_x, 0.0],
                [0.0, process_noise_v]
            ],
            dtype=np.float32
        )

        self.Q = self.Q_base.copy()

        # 基础R
        self.R_base = float(measure_noise)

    def reset(self):
        """
        清空Track
        """
        self.initialized = False
        self.x[:] = 0
        self.P[:] = np.eye(2, dtype=np.float32)
        self.Q[:] = self.Q_base

    def init(self, x):
        self.initialized = True
        self.x[0, 0] = float(x)
        self.x[1, 0] = 0.0
        self.P[:] = np.eye(2, dtype=np.float32)

    def predict(self):
        """
        预测
        """
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def update(self, z, score):
        """
        更新

        Parameters
        ----------
        z : float
            当前检测x

        score : float
            当前点置信度
        """
        if not self.initialized:
            self.init(z)
            return float(z)

        # score限制
        score = float(np.clip(score, 0.05, 1.0))

        # score越低
        # 越相信预测
        R = self.R_base / score

        # score越低
        # 允许预测变化更大
        q_scale = 1.0 + (1.0 - score) * 2.0

        self.Q[0, 0] = self.Q_base[0, 0] * q_scale
        self.Q[1, 1] = self.Q_base[1, 1] * q_scale

        # innovation
        innovation = z - (self.H @ self.x)[0, 0]

        # innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman Gain
        K = self.P @ self.H.T / S

        # 更新状态
        self.x = self.x + K * innovation

        # 更新协方差
        I = np.eye(2, dtype=np.float32)

        self.P = (I - K @ self.H) @ self.P

        return float(self.x[0, 0])

    @property
    def position(self):
        """
        当前x
        """
        if not self.initialized:
            return None
        return float(self.x[0, 0])

    @property
    def velocity(self):
        """
        当前速度
        """
        if not self.initialized:
            return 0.0
        return float(self.x[1, 0])

class LaneTracker:
    """
    固定4条车道 × 固定58个row 的车道跟踪器
    """
    def __init__(self,lane_num=4,point_num=58,
                # 初始化score
                init_score_thresh=0.80,

                # 正常更新score
                score_thresh=0.60,

                # Gate参数
                gate_base=8.0,
                gate_scale=0.08,

                # 连续丢失多少帧删除
                max_miss=5):

        self.lane_num = lane_num
        self.point_num = point_num

        self.init_score_thresh = init_score_thresh
        self.score_thresh = score_thresh

        self.gate_base = gate_base
        self.gate_scale = gate_scale

        self.max_miss = max_miss

        # Kalman
        self.filters = []

        # Track状态
        self.track_state = []

        for _ in range(lane_num):
            lane_filter = []
            lane_state = []
            for _ in range(point_num):
                lane_filter.append(Kalman1D())
                lane_state.append({"miss": 0,})

            self.filters.append(lane_filter)
            self.track_state.append(lane_state)

    def reset(self):
        for i in range(self.lane_num):
            for j in range(self.point_num):
                self.filters[i][j].reset()
                self.track_state[i][j]["miss"] = 0

    def update(self, coords, scores):
        """
        Parameters
        ----------
        coords

        [
            lane0:[(x,y),...],
            lane1...
        ]

        scores

        [
            lane0:[score...],
            lane1...
        ]
        """
        output = []
        for lane_idx in range(self.lane_num):
            lane = coords[lane_idx]
            lane_scores = scores[lane_idx]

            # 建立查询表
            point_dict = {}
            score_dict = {}
            for p, s in zip(lane, lane_scores):
                point_dict[int(p[1])] = float(p[0])
                score_dict[int(p[1])] = float(s)

            new_lane = []

            # 固定58个row
            for row_idx, y in enumerate(ROW_COORDS):
                kf = self.filters[lane_idx][row_idx]
                state = self.track_state[lane_idx][row_idx]

                # 未初始化
                if not kf.initialized:
                    if y not in point_dict:
                        continue
                    detect_x = point_dict[y]
                    score = score_dict[y]
                    if score >= self.init_score_thresh:
                        kf.init(detect_x)
                        state["miss"] = 0
                        new_lane.append((detect_x, y))
                    continue

                # predict
                pred_x = kf.predict()

                # 当前row没有检测
                if y not in point_dict:
                    if kf.initialized:
                        state["miss"] += 1
                        if state["miss"] > self.max_miss:
                            kf.reset()
                            state["miss"] = 0
                        else:
                            new_lane.append((pred_x, y))
                    continue

                # score太低
                if score_dict[y] < self.score_thresh:
                    state["miss"] += 1
                    if state["miss"] > self.max_miss:
                        kf.reset()
                        state["miss"] = 0
                    else:
                        new_lane.append((pred_x, y))
                    continue

                # 动态Gate
                gate = self.gate_base + self.gate_scale * y

                # score越高
                # Gate越宽
                gate *= score_dict[y]

                # Gate失败
                if abs(point_dict[y] - pred_x) > gate:
                    state["miss"] += 1
                    if state["miss"] > self.max_miss:
                        kf.reset()
                        state["miss"] = 0
                    else:
                        new_lane.append((pred_x, y))
                    continue

                # Kalman Update
                x = kf.update(point_dict[y], score_dict[y])
                state["miss"] = 0
                new_lane.append((x, y))
            output.append(new_lane)
        return output

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
    """
    image_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    for file_path in glob.glob(os.path.join(data_root, "**", "*"), recursive=True):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_extensions:
            image_paths.append(file_path)
    print(f"✅ 共找到 {len(image_paths)} 张图片")
    return image_paths

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ONNX车道线检测推理')
    parser.add_argument('--mode', type=str, default='video', choices=['video'], help='运行模式: json=保存坐标JSON, video=生成视频, eval=计算测试集Loss')
    parser.add_argument('--onnx_model', type=str, default="../model/output_model.onnx", help='ONNX模型文件路径')
    parser.add_argument('--demo_data_root', type=str, default="../../../data/dispose/images", help='图片目录路径')
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
    pred = {}
    for name, val in zip(output_names, outputs):
        pred[name] = val
    return pred
def find_output_key(pred, target_key):
    """
    从pred字典中查找包含target_key的输出
    支持输出名称带前缀的情况，如 'output_loc_row' 匹配 'loc_row'
    """
    if target_key in pred:
        return pred[target_key]
    for k, v in pred.items():
        if target_key in k:
            return v
    raise KeyError(f"找不到输出键: {target_key}, 可用键: {list(pred.keys())}")
def sigmoid_np(x):
    """numpy实现sigmoid"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))
def softmax_np(x):
    """numpy实现softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
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

    # ============ 对 exist_row 做 softmax，转换为概率 ============
    # 数值稳定版 softmax（防止 exp 溢出）
    exist_max = np.max(exist_row, axis=1, keepdims=True)          # (B, 1, 58, 4)
    exp_exist = np.exp(exist_row - exist_max)                     # (B, 2, 58, 4)
    exist_softmax = exp_exist / np.sum(exp_exist, axis=1, keepdims=True)  # (B, 2, 58, 4)

    # ============  取出"存在"（index=1）的概率 ============
    exist_prob = exist_softmax[:, 1, :, :]                        # (B, 58, 4)
    # 每个值 ∈ [0, 1]，表示该锚点存在车道线的概率

    # ============ 用 valid_row 做掩码，只看存在的点 ============
    # valid_row 已经是 (B, 58, 4)，值为 0 或 1
    valid_prob = exist_prob * valid_row                            # (B, 58, 4)
    # 不存在的点概率被置零，存在的点保留其概率值

    # 获取车道线属性标签
    lane_label_raw = find_output_key(pred, 'lane_label')
    lane_label_prob = sigmoid_np(lane_label_raw)  # (B, 4, 8)
    lane_label = (lane_label_prob > 0.5).astype(np.uint8)
    # 预分配结果
    all_coords = [[] for _ in range(batch_size)]
    all_lane_labels = [{} for _ in range(batch_size)]
    all_scores = [[] for _ in range(batch_size)]
    row_lane_idx = [0, 1, 2, 3]
    for b in range(batch_size):
        img_w = original_image_widths[b]
        img_h = original_image_heights[b]
        coords = []
        lane_labels = {}
        scores = []
        for i in row_lane_idx:
            tmp = []
            tmp_scores = []
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
                        tmp_scores.append(valid_prob[b, k, i])
            coords.append(tmp)
            scores.append(tmp_scores)
            for tmp_n in range(len(lane_label_index_dic)):
                if lane_label[b, i, tmp_n]:
                    if str(i) not in lane_labels:
                        lane_labels[str(i)] = []
                    lane_labels[str(i)].append(lane_label_index_dic[tmp_n])

        all_coords[b] = coords
        all_lane_labels[b] = lane_labels
        all_scores[b] = scores
    return all_coords, all_lane_labels, all_scores
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
    for num, lane in enumerate(filter_coords):
        if len(lane) == 0:
            continue
        for num1, coord in enumerate(lane):
            # if num == 0:
            #     cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 0, 0), -1)
            # elif num == 1:
            #     cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 255, 0), -1)
            # elif num == 2:
            #     cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 0, 0), -1)
            # else:
            #     cv2.circle(vis, (int(coord[0]), int(coord[1])), 8, (0, 255, 0), -1)
            
            if num1 == 0:
                continue
            cv2.line(vis, (int(lane[num1 - 1][0]), int(lane[num1 - 1][1])) , (int(lane[num1][0]), int(lane[num1][1])),  (0, 255, 0), 7)
    # 绘制标签文字（使用PIL支持中文）
    sorted_keys = sorted(lane_labels.keys(), key=lambda x: int(x))
    for tmp_i, lane_index in enumerate(sorted_keys):
        label = lane_labels[lane_index]
        draw_point = (20, 40 + tmp_i * 80)
        vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_pil)
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

def lane_cut_cross(all_lane_labels, all_p):
    '''
    '''
    result_lane_labels = []
    for b in range(len(all_lane_labels)):
        # 找lane_coords所有车道最小y值
        start_y = []
        for j in range(len(all_lane_labels[b])):
            lane = all_lane_labels[b][j]
            if len(lane) == 0:
                continue
            # 按y值排序 从小到大排序
            lane = lane[lane[:, 1].argsort()]
            start_y.append(lane[0][1])
        start_y = int(min(start_y))

        if (start_y >= 2160 or start_y < 0 or len(all_lane_labels[b]) < 2):
            result_lane_labels.append(all_lane_labels[b])
            continue
        
        cut_y = -1.0
        for temp_y in range(start_y, 2160, 2):
            last_x = -99999.0
            valid_lane_num = 0
            valid = True
            for i in range(len(all_lane_labels[b])):
                lane = all_lane_labels[b][i]
                if len(lane) == 0:
                    continue
                p = all_p[b][i]

                if (lane[:, 1] < temp_y).any():
                    top = lane[lane[:, 1] < temp_y][0]
                    bottom = lane[lane[:, 1] >= temp_y][0]
                    current_x = (top[0] - bottom[0]) / (top[1] - bottom[1]) * (temp_y - bottom[1]) + bottom[0]
                else:
                    current_x = 0
                    for k in range(len(p)):
                        current_x = current_x * temp_y + p[k]
                
                if valid_lane_num > 0 and current_x <= last_x:
                    valid = False
                    break
                
                last_x = current_x
                valid_lane_num += 1

            if valid and valid_lane_num >= 2:
                cut_y = temp_y
                break
        
        if (cut_y < 0.0):
            result_lane_labels.append(all_lane_labels[b])
            continue

        # 裁剪
        temp_save = []
        for m in range(len(all_lane_labels[b])):
            lane = all_lane_labels[b][m]
            if len(lane) == 0:
                temp_save.append([])
                continue
            
            # 裁剪
            lane = lane[lane[:, 1] > cut_y]
            temp_save.append(lane)
        
        result_lane_labels.append(temp_save)
    return result_lane_labels

def lane_filter(lane_coords):
    '''
    '''
    #  y 坐标对点集进行排序从小到大 
    lane_coords = lane_coords[lane_coords[:, 1].argsort()]

    # 取前6个点
    filtered = lane_coords[:6]
    x = filtered[:, 0]
    y = filtered[:, 1]

    # 做一次拟合
    p = np.polyfit(y, x, 2)

    top_y = 360 * 0.3
    min_x = 640 * 0.002     # 左边界，0.2%位置
    max_x = 640 * 0.998    # 右边界，99.8%位置
    step = 2.0
    new_coords = []
    if (top_y < y[-1]):
        current_y = y[-1] - step
        
        while (current_y >= top_y):
            current_x = 0.0

            for k in range(len(p)):
                current_x = current_x * current_y + p[k]
            
            # 检查边界：x坐标范围
            if (current_x < min_x or current_x > max_x):
                break;
            
            new_coords.append((current_x, current_y))
            
            current_y -= step

    # lane_coords去掉前6个点，并且在后面追加new_coords的点
    lane_coords = np.concatenate((lane_coords[6:], np.array(new_coords)), axis=0)

    # y坐标从小到大
    lane_coords = lane_coords[lane_coords[:, 1].argsort()]

    tck, u = splprep([lane_coords[:, 0], lane_coords[:, 1]], s=5.0)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = splev(u_new, tck)
    lane_coords = np.column_stack((x_new, y_new))
        
    return lane_coords, p

def lane_post_optimize(all_coords, original_widths, original_heights, input_width, input_height):
    '''
    '''
    all_lane_labels = []
    all_p = []
    # 去除异常点
    for b in range(len(all_coords)):
        temp_all_lane_labels = []
        temp_p = []
        for i in range(len(all_coords[b])):
            lane = all_coords[b][i]
            if len(lane) < 10:
                temp_all_lane_labels.append([])
                temp_p.append([])
                continue
            
            pts = np.array(lane, dtype=np.float32)
            filtered = []
            
            # 越界过滤：x坐标超出图像范围的点剔除
            for p in pts:
                if 0 <= p[0] <= input_width and 0 <= p[1] <= input_height:
                    filtered.append(p)
            filtered = np.array(filtered)

            if len(filtered) < 5:
                temp_all_lane_labels.append(filtered)
                temp_p.append((0.0,0.0,1.0))
                continue

            filtered, p = lane_filter(filtered)

            temp_all_lane_labels.append(filtered)
            temp_p.append(p)

        all_lane_labels.append(temp_all_lane_labels)
        all_p.append(temp_p) 
    
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
    
    for b in range(len(all_lane_labels)):
        scale_x = original_widths[b] / input_width
        scale_y = original_heights[b] / input_height

        for i in range(len(all_lane_labels[b])):
            lane = all_lane_labels[b][i]
            if len(lane) == 0:
                continue
            pts = np.array(lane, dtype=np.float32)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            all_lane_labels[b][i] = pts

            one_p = all_p[b][i]
            one_p[0] = one_p[0] / scale_y
            one_p[1] = one_p[1]
            one_p[2] = one_p[2] * scale_x
            all_p[b][i] = one_p

    # 交叉坐标过滤
    all_lane_labels = lane_cut_cross(all_lane_labels, all_p)

    return all_coords, all_lane_labels

def run_video_mode(session, input_name, output_names, image_list, args):
    """视频模式：将带车道线的图片合成视频"""
    batch_size = args.batch_size 
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    image_list = natsorted(image_list, key=os.path.basename)
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

    # 初始化跟踪器
    tracker = LaneTracker()

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
        all_coords, all_lane_labels, all_scores = pred2coords(pred, original_image_widths=original_widths, original_image_heights=original_heights, train_width=args.train_width, resize_mode='train')
        # 跟踪
        for b in range(batch_size):
            all_coords[b] = tracker.update(all_coords[b],all_scores[b])

        # 后处理
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
def main():
    args = parse_args()
    session, input_name, output_names = load_onnx_model(args.onnx_model)
    image_list = collect_images(data_root=args.demo_data_root, mode=args.mode, test_txt_path=args.test_txt)
    if len(image_list) == 0:
        print("未找到任何图片，退出。")
        return
    run_video_mode(session, input_name, output_names, image_list, args)
if __name__ == "__main__":
    main()