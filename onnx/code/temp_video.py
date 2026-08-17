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
from scipy.interpolate import splprep, splev, PchipInterpolator
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.signal import medfilt
from sklearn.neighbors import LocalOutlierFactor
ROW_COORDS = [126, 128, 130, 132, 134, 137, 139, 142, 144, 147,
              149, 152, 155, 158, 161, 164, 167, 170, 173, 176,
              180, 183, 187, 190, 194, 197, 201, 204, 208, 212,
              216, 220, 224, 228, 233, 237, 242, 246, 251, 256,
              261, 266, 271, 276, 282, 287, 293, 298, 304, 310,
              316, 322, 328, 335, 341, 347, 353, 359]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
lane_label_index_dic = ["单", "多", "白", "黄", "实线", "虚线", "实虚线", "虚实线"]

XMEDIA_SVP_LANE_MAX_NUM = 4 # 最大车道数
XMEDIA_SVP_LANE_MAX_ROW_POINT = 58 # 每个车道的row点数
XMEDIA_SVP_LANE_MAX_FITTED_POINT = 360 # 640x360坐标系中的稠密点最大数量
XMEDIA_SVP_LANE_LABEL_NUM = 8 # 车道线类型数量
LANE_SPLINE_COEF_NUM = 8 # 增加局部自由度，以适应急弯和复合弯道

LANE_CURVE_INVALID = 0
LANE_CURVE_STRAIGHT = 1
LANE_CURVE_SPLINE = 2

LANE_FIT_MIN_POINT_NUM = 7
LANE_POINT_MIN_SCORE = 0.50
LANE_TRACK_MAX_MISS = 3
HOMOGRAPHY_MIN_DENOMINATOR = 0.08
LANE_SPLINE_SMOOTH_LAMBDA = 0.0001
LANE_EXTENSION_MIN_POINT_NUM = 12
LANE_FAR_TRIM_MIN_POINT_NUM = 8
LANE_FAR_TRIM_MIN_RAW_SLOPE = 0.03
LANE_FAR_TRIM_ANALYSIS_SPAN = 12.0
LANE_FAR_TRIM_CONFIRM_SPAN = 24.0
LANE_FAR_TRIM_MAX_LENGTH = 10.0
LANE_FAR_TRIM_MIN_EXCURSION = 0.50
LANE_FAR_TRIM_MIN_RECOVERY = 1.00

# 拟合质量阈值。模型坐标阈值的单位均为640x360图上的像素。
LANE_STRAIGHT_MAX_MODEL_RMS = 1.5
LANE_STRAIGHT_MAX_MODEL_P90 = 3.5
LANE_FIT_MAX_BEV_RMS = 12.0
LANE_FIT_MAX_MODEL_RMS = 6.0
LANE_FIT_MAX_MODEL_P90 = 12.0

# 时序门控：普通变化立即更新；单线大变化需要连续高质量测量确认。
LANE_TRACK_FAST_UPDATE_MODEL_RMS = 6.0
LANE_TRACK_ACCEPT_MODEL_RMS = 18.0
LANE_TRACK_PENDING_SHAPE_RMS = 8.0
LANE_TRACK_PENDING_MIN_DIRECTION = 5.0
LANE_TRACK_PENDING_MAX_AGE = 3
LANE_ORDER_REJECT_ERROR_MARGIN = 3.0

# 两条内线相邻帧共同运动：用于区分真实大弯与单条车道突跳。
LANE_DUAL_MOTION_SAMPLE_NUM = 16
LANE_DUAL_MOTION_MIN_POINT_NUM = 24
LANE_DUAL_MOTION_MIN_BEV_OVERLAP = 40.0
LANE_DUAL_MOTION_MAX_FIT_RMS = 3.0
LANE_DUAL_MOTION_MAX_FIT_P90 = 5.0
LANE_DUAL_MOTION_MIN_MEDIAN_DX = 8.0
LANE_DUAL_MOTION_MIN_LANE_SIGN_RATIO = 0.75
LANE_DUAL_MOTION_MIN_PAIR_SIGN_RATIO = 0.80
LANE_DUAL_MOTION_MIN_MAG_RATIO = 0.30
LANE_DUAL_MOTION_MAX_PAIR_DIFF_RMS = 30.0

LANE_TRACK_STATUS_INVALID = 0
LANE_TRACK_STATUS_MEASURED = 1
LANE_TRACK_STATUS_PREDICTED = 2
LANE_TRACK_STATUS_PENDING = 3

LANE_POINT_NUM = [0]*XMEDIA_SVP_LANE_MAX_NUM
ROW_MASK = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT), dtype=np.int32)
ROW_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT, 3), dtype=np.float32)

LANE_FITTED_POINT_NUM = [0]*XMEDIA_SVP_LANE_MAX_NUM
ROW_FITTED_MASK = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_FITTED_POINT), dtype=np.int32)
ROW_FITTED_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_FITTED_POINT, 3), dtype=np.float32)

LANE_LABEL_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_LABEL_NUM), dtype=np.int32)

# 单帧BEV中间结果，与模型原始输出分开，方便逐步调试。
BEV_POINT_NUM = [0] * XMEDIA_SVP_LANE_MAX_NUM
BEV_MASK = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT), dtype=np.int32)
BEV_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT, 3), dtype=np.float32)

BEV_FILTERED_POINT_NUM = [0] * XMEDIA_SVP_LANE_MAX_NUM
BEV_FILTERED_MASK = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT), dtype=np.int32)
BEV_FILTERED_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_ROW_POINT, 3), dtype=np.float32)

BEV_FITTED_POINT_NUM = [0] * XMEDIA_SVP_LANE_MAX_NUM
BEV_FITTED_MASK = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_FITTED_POINT), dtype=np.int32)
BEV_FITTED_RESULT = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, XMEDIA_SVP_LANE_MAX_FITTED_POINT, 3), dtype=np.float32)

# 当前帧拟合结果：B样条只拟合 x=f(y)，y 使用固定BEV范围归一化。
LANE_CURVE_TYPE = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
LANE_SPLINE_COEF = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, LANE_SPLINE_COEF_NUM), dtype=np.float32)
LANE_SPLINE_Y_RANGE = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, 2), dtype=np.float32)
LANE_PROCESSED_MODEL_LENGTH = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_EXTENSION_BEV_LENGTH = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_FIT_MODEL_RMS = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_FIT_MODEL_P90 = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_TRACK_MODEL_RMS = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_TRACK_STATUS = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
LANE_TOPOLOGY_REJECTED = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
LANE_COMMON_MOTION_ACCEPTED = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
LANE_OBSERVED_MODEL_SPAN = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_CURVE_FAR_DEVIATION = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
LANE_FAR_TRIMMED_POINT_NUM = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)

# 跨帧跟踪状态，reset_frame_results()不能清空这些变量。
TRACKER_INITIALIZED = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
TRACKER_STATE = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, LANE_SPLINE_COEF_NUM, 2), dtype=np.float32)
TRACKER_P = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, LANE_SPLINE_COEF_NUM, 2, 2), dtype=np.float32)
TRACKER_MISS = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
TRACKER_Y_RANGE = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, 2), dtype=np.float32)
TRACKER_CONFIDENCE = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
TRACKER_CURVE_TYPE = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
TRACKER_TYPE_SWITCH_COUNT = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)

# 大创新候选：只在连续两帧的新曲线相互一致时，才替换当前可靠轨迹。
TRACKER_PENDING_COEF = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, LANE_SPLINE_COEF_NUM), dtype=np.float32)
TRACKER_PENDING_Y_RANGE = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, 2), dtype=np.float32)
TRACKER_PENDING_CONFIDENCE = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)
TRACKER_PENDING_CURVE_TYPE = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
TRACKER_PENDING_COUNT = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
TRACKER_PENDING_DIRECTION = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.float32)

# 上一帧完整当前测量，仅用于识别双内线共同运动，不等同于跟踪输出。
PREV_MEASUREMENT_VALID = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)
PREV_MEASUREMENT_COEF = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, LANE_SPLINE_COEF_NUM), dtype=np.float32)
PREV_MEASUREMENT_Y_RANGE = np.zeros((XMEDIA_SVP_LANE_MAX_NUM, 2), dtype=np.float32)
PREV_MEASUREMENT_POINT_NUM = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=np.int32)


# =============================================================================
# 后处理阅读指南（建议先看 lane_post_optimize，再按下面顺序跳到各函数）
# =============================================================================
#
# 模型输出的并不是一条连续曲线，而是：每条车道在58个固定行锚点上的
# “横坐标x + 存在概率”。pred2coords()把它们写入 ROW_*，后处理再把
# 这些稀疏且可能抖动/漏检的点，变成最终可绘制的稳定稠密曲线。
#
# 单帧数据流：
#
#   ROW_RESULT / ROW_MASK                 模型坐标(640x360)中的稀疏原始点
#       |  单应变换H：透视图 -> 鸟瞰图
#       v
#   BEV_RESULT / BEV_MASK                 BEV中的原始点
#       |  置信度过滤 + 局部趋势飞点过滤
#       v
#   BEV_FILTERED_RESULT / MASK            BEV中的可靠测量点
#       |  先尝试直线，必要时拟合8控制量三次B样条
#       v
#   LANE_SPLINE_COEF                      本帧曲线测量
#       |  车道左右顺序校验 + 跨帧跟踪/短时预测
#       v
#   BEV_FITTED_RESULT                     跟踪后生成的BEV稠密点
#       |  逆单应变换H_inv + 原图尺寸缩放
#       v
#   ROW_FITTED_RESULT / MASK              最终原图坐标，供draw_fitted_lanes绘制
#
# 为什么先转BEV再拟合：透视图中的平行车道会向消失点收敛，同一条真实
# 直线在图上可能强烈弯折；BEV近似俯视路面，x=f(y)更容易拟合、判断
# 左右顺序和做远端延长。为什么误差又要回到640x360判断：BEV靠近地平线
# 的尺度会被放大，固定BEV像素阈值不稳定，而最终肉眼看到的是图像误差。
#
# 两类状态的生命周期一定要区分：
#   1. reset_frame_results() 每帧调用，清空ROW/BEV/最终绘制等单帧缓存；
#   2. reset_lane_trackers() 每段新视频只调用一次，保留相邻帧之间的连续性。
# 如果误把跟踪器每帧清零，就只剩单帧拟合；如果不清单帧缓存，上一帧点会
# 混进当前帧，产生并不存在的车道线。
# =============================================================================


def reset_frame_results():
    """清空单帧输出缓存，避免上一帧的检测结果残留。"""
    # 原地清零，pred2coords/draw_lane_on_image 仍然访问同一组全局对象。
    # 对应到 C 语言中就是在处理新帧前对结果结构体/数组执行 memset(0)。
    LANE_POINT_NUM[:] = [0] * XMEDIA_SVP_LANE_MAX_NUM
    LANE_FITTED_POINT_NUM[:] = [0] * XMEDIA_SVP_LANE_MAX_NUM
    ROW_MASK.fill(0)
    ROW_RESULT.fill(0)
    ROW_FITTED_MASK.fill(0)
    ROW_FITTED_RESULT.fill(0)
    LANE_LABEL_RESULT.fill(0)
    BEV_POINT_NUM[:] = [0] * XMEDIA_SVP_LANE_MAX_NUM
    BEV_MASK.fill(0)
    BEV_RESULT.fill(0)
    BEV_FILTERED_POINT_NUM[:] = [0] * XMEDIA_SVP_LANE_MAX_NUM
    BEV_FILTERED_MASK.fill(0)
    BEV_FILTERED_RESULT.fill(0)
    BEV_FITTED_POINT_NUM[:] = [0] * XMEDIA_SVP_LANE_MAX_NUM
    BEV_FITTED_MASK.fill(0)
    BEV_FITTED_RESULT.fill(0)
    LANE_CURVE_TYPE.fill(LANE_CURVE_INVALID)
    LANE_SPLINE_COEF.fill(0)
    LANE_SPLINE_Y_RANGE.fill(0)
    LANE_PROCESSED_MODEL_LENGTH.fill(0)
    LANE_EXTENSION_BEV_LENGTH.fill(0)
    LANE_FIT_MODEL_RMS.fill(0)
    LANE_FIT_MODEL_P90.fill(0)
    LANE_TRACK_MODEL_RMS.fill(0)
    LANE_TRACK_STATUS.fill(LANE_TRACK_STATUS_INVALID)
    LANE_TOPOLOGY_REJECTED.fill(0)
    LANE_COMMON_MOTION_ACCEPTED.fill(0)
    LANE_OBSERVED_MODEL_SPAN.fill(0)
    LANE_CURVE_FAR_DEVIATION.fill(0)
    LANE_FAR_TRIMMED_POINT_NUM.fill(0)


def reset_lane_trackers():
    """清空跨帧跟踪器，只在开始新视频时调用。"""
    TRACKER_INITIALIZED.fill(0)
    TRACKER_STATE.fill(0)
    TRACKER_P.fill(0)
    TRACKER_MISS.fill(0)
    TRACKER_Y_RANGE.fill(0)
    TRACKER_CONFIDENCE.fill(0)
    TRACKER_CURVE_TYPE.fill(LANE_CURVE_INVALID)
    TRACKER_TYPE_SWITCH_COUNT.fill(0)
    TRACKER_PENDING_COEF.fill(0)
    TRACKER_PENDING_Y_RANGE.fill(0)
    TRACKER_PENDING_CONFIDENCE.fill(0)
    TRACKER_PENDING_CURVE_TYPE.fill(LANE_CURVE_INVALID)
    TRACKER_PENDING_COUNT.fill(0)
    TRACKER_PENDING_DIRECTION.fill(0)
    PREV_MEASUREMENT_VALID.fill(0)
    PREV_MEASUREMENT_COEF.fill(0)
    PREV_MEASUREMENT_Y_RANGE.fill(0)
    PREV_MEASUREMENT_POINT_NUM.fill(0)

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

def collect_images(data_root):
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
    parser.add_argument('--onnx_model', type=str, default="../model/output_model.onnx", help='ONNX模型文件路径')
    parser.add_argument('--demo_data_root', type=str, default="../../../data/dispose/images2", help='图片目录路径')
    parser.add_argument('--train_height', type=int, default=288, help='模型输入高度')
    parser.add_argument('--train_width', type=int, default=640, help='模型输入宽度')
    parser.add_argument('--crop_ratio', type=float, default=0.8, help='裁剪比例')
    parser.add_argument('--save_path', type=str, default='./test_results/', help='结果保存路径')
    parser.add_argument('--fps', type=float, default=1.0, help='视频帧率')
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

def pred2coords(pred, local_width=5, train_width=640):
    """
    预测结果转坐标
    :param pred: ONNX输出的字典
    :param train_width: 模型输入宽度
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

    row_lane_idx = [0, 1, 2, 3]
    for b in range(batch_size):
        for i in row_lane_idx:
            tmp = []
            if valid_row[b, :, i].sum() > 0:
                for k in range(valid_row.shape[1]):
                    if valid_row[b, k, i]:
                        idx_center = int(max_indices_row[b, k, i])
                        all_ind = list(range(max(0, idx_center - local_width), min(num_grid_row - 1, idx_center + local_width) + 1))
                        all_ind = np.array(all_ind, dtype=np.int64)

                        raw_vals = loc_row[b, all_ind, k, i]  # (local_width*2+1,)
                        probs = softmax_np(raw_vals)
                        out_tmp = np.sum(probs * all_ind.astype(np.float32))
                        
                        out_tmp = out_tmp / (num_grid_row - 1) * train_width
                        y_val = ROW_COORDS[k]
                        tmp.append((k, out_tmp, y_val, valid_prob[b, k, i]))
            if len(tmp) <= 6:
                continue

            LANE_POINT_NUM[i] = len(tmp)
            for coord in tmp:
                ROW_RESULT[i, int(coord[0])] = coord[1], coord[2], coord[3]
                ROW_MASK[i, int(coord[0])] = 1

            for tmp_n in range(len(lane_label_index_dic)):
                if lane_label[b, i, tmp_n]:
                    LANE_LABEL_RESULT[i, tmp_n] = 1


def weighted_line_fit(points, robust_iterations=3):
    """使用置信度和Huber权重拟合 x=a*y+b。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return None, None

    x = points[:, 0]
    y = points[:, 1]
    base_weight = np.clip(points[:, 2], 0.05, 1.0)
    y_center = float(np.average(y, weights=base_weight))
    design = np.column_stack((y - y_center, np.ones(len(y))))
    weight = base_weight.copy()
    solution = None

    for _ in range(max(1, robust_iterations)):
        normal = design.T @ (weight[:, None] * design)
        rhs = design.T @ (weight * x)
        try:
            solution = np.linalg.solve(normal + np.eye(2) * 1e-6, rhs)
        except np.linalg.LinAlgError:
            return None, None

        residual = x - design @ solution
        median = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - median))
        if sigma < 1e-3:
            break
        huber_limit = 2.5 * sigma
        robust_weight = np.ones(len(residual), dtype=np.float64)
        large = np.abs(residual) > huber_limit
        robust_weight[large] = huber_limit / np.abs(residual[large])
        weight = base_weight * robust_weight

    slope = float(solution[0])
    intercept = float(solution[1] - slope * y_center)
    residual = x - (slope * y + intercept)
    return np.array([slope, intercept], dtype=np.float32), residual.astype(np.float32)


def filter_bev_lane_points(lane_idx, window_radius=2):
    """
    按置信度和局部直线趋势删除BEV飞点。

    这里不拿“整条车道的一条直线”过滤，因为真实车道可能是弯道。对每个
    点只查看前后少量邻居，用局部趋势预测当前x；偏差大于
    max(15 BEV像素, 4倍局部波动)才删除，因此能去掉孤立飞点，又尽量
    保留连续弯曲。返回值每行仍是[x_bev, y_bev, score]。
    """
    valid_indices = np.flatnonzero(BEV_MASK[lane_idx])
    if len(valid_indices) == 0:
        return np.empty((0, 3), dtype=np.float32)

    points = BEV_RESULT[lane_idx, valid_indices].copy()
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite & (points[:, 2] >= LANE_POINT_MIN_SCORE)]
    if len(points) == 0:
        return points

    points = points[np.argsort(points[:, 1])]
    if len(points) < 5:
        return points

    keep = np.ones(len(points), dtype=bool)
    for point_idx in range(len(points)):
        start = max(0, point_idx - window_radius)
        end = min(len(points), point_idx + window_radius + 1)
        neighbor_indices = [idx for idx in range(start, end) if idx != point_idx]
        if len(neighbor_indices) < 3:
            continue

        neighbors = points[neighbor_indices]
        line, neighbor_residual = weighted_line_fit(neighbors, robust_iterations=1)
        if line is None:
            continue
        predicted_x = line[0] * points[point_idx, 1] + line[1]
        local_sigma = 1.4826 * np.median(np.abs(neighbor_residual - np.median(neighbor_residual)))
        reject_threshold = max(15.0, 4.0 * local_sigma)
        if abs(points[point_idx, 0] - predicted_x) > reject_threshold:
            keep[point_idx] = False

    return points[keep]


def save_filtered_bev_lane(lane_idx, points):
    """保存过滤后的BEV点，方便单独查看该阶段。"""
    point_num = min(len(points), XMEDIA_SVP_LANE_MAX_ROW_POINT)
    if point_num == 0:
        return
    BEV_FILTERED_RESULT[lane_idx, :point_num] = points[:point_num]
    BEV_FILTERED_MASK[lane_idx, :point_num] = 1
    BEV_FILTERED_POINT_NUM[lane_idx] = point_num


def build_bspline_basis(normalized_y, control_num=LANE_SPLINE_COEF_NUM, degree=3):
    """
    构建固定开区间均匀三次B样条基矩阵B，最终曲线满足 x = B @ coef。

    所有帧都使用同一个BEV定义域和8个系数，因此“第i个系数”跨帧含义
    一致，后面的卡尔曼跟踪才能逐系数进行；若每帧按自身可见范围归一化，
    同一组系数会代表不同道路位置，数值便无法直接比较或平滑。
    """
    normalized_y = np.clip(np.asarray(normalized_y, dtype=np.float64), 0.0, 1.0)
    interior = np.linspace(0.0, 1.0, control_num - degree + 1)[1:-1]
    knots = np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))
    memo = {}

    def basis(control_idx, current_degree):
        key = (control_idx, current_degree)
        if key in memo:
            return memo[key]
        if current_degree == 0:
            value = ((normalized_y >= knots[control_idx]) &
                     (normalized_y < knots[control_idx + 1])).astype(np.float64)
            if control_idx == control_num - 1:
                value[normalized_y == 1.0] = 1.0
            memo[key] = value
            return value

        value = np.zeros_like(normalized_y)
        left_denominator = knots[control_idx + current_degree] - knots[control_idx]
        right_denominator = knots[control_idx + current_degree + 1] - knots[control_idx + 1]
        if left_denominator > 0:
            value += ((normalized_y - knots[control_idx]) / left_denominator) * basis(control_idx, current_degree - 1)
        if right_denominator > 0:
            value += ((knots[control_idx + current_degree + 1] - normalized_y) / right_denominator) * basis(control_idx + 1, current_degree - 1)
        memo[key] = value
        return value

    return np.column_stack([basis(idx, degree) for idx in range(control_num)])


def normalize_bev_y(y, bev_y_domain):
    """把BEV y映射到跨帧固定的[0, 1]区间。"""
    y_min, y_max = bev_y_domain
    span = max(y_max - y_min, 1e-6)
    return (np.asarray(y, dtype=np.float64) - y_min) / span


def evaluate_bspline(coefficients, y, bev_y_domain):
    """计算固定系数B样条的 x=f(y)。"""
    basis = build_bspline_basis(normalize_bev_y(y, bev_y_domain))
    return basis @ np.asarray(coefficients, dtype=np.float64)


def calculate_curve_model_point_errors(coefficients, points, bev_y_domain, H_inv):
    """
    计算曲线与BEV点逐点对应的误差，并换算到640x360模型坐标。

    BEV在地平线附近的尺度变化很大，不能直接用固定BEV像素阈值做
    跨帧判断；逆变换后的模型像素误差更稳定，也更符合最终画线偏差。
    无法安全逆变换的点返回inf，调用者可以直接排除。
    """
    points = np.asarray(points, dtype=np.float64)
    errors = np.full(len(points), np.inf, dtype=np.float64)
    if coefficients is None or len(points) == 0:
        return errors

    predicted_x = evaluate_bspline(coefficients, points[:, 1], bev_y_domain)
    predicted_bev = np.column_stack((predicted_x, points[:, 1]))
    predicted_model, predicted_valid = perspective_transform_points(predicted_bev, H_inv)
    measured_model, measured_valid = perspective_transform_points(points[:, :2], H_inv)
    valid = predicted_valid & measured_valid
    if np.any(valid):
        errors[valid] = np.linalg.norm(
            predicted_model[valid] - measured_model[valid], axis=1
        )
    return errors


def calculate_curve_fit_metrics(coefficients, points, bev_y_domain, H_inv):
    """返回曲线在模型坐标中的加权RMS和P90误差。"""
    if coefficients is None or points is None or len(points) == 0:
        return float("inf"), float("inf")
    errors = calculate_curve_model_point_errors(
        coefficients, points, bev_y_domain, H_inv
    )
    valid = np.isfinite(errors)
    if not np.any(valid):
        return float("inf"), float("inf")
    weights = np.clip(np.asarray(points)[valid, 2], 0.05, 1.0)
    rms = float(np.sqrt(np.average(errors[valid] ** 2, weights=weights)))
    p90 = float(np.percentile(errors[valid], 90))
    return rms, p90


def evaluate_curve_in_model(coefficients, sample_y, bev_y_domain, H_inv):
    """在给定BEV y位置计算曲线对应的640x360模型坐标。"""
    sample_y = np.asarray(sample_y, dtype=np.float64)
    sample_x = evaluate_bspline(coefficients, sample_y, bev_y_domain)
    model_points, valid = perspective_transform_points(
        np.column_stack((sample_x, sample_y)), H_inv
    )
    return model_points, valid


def calculate_curve_motion_metrics(current_coefficients, current_y_range,
                                   reference_coefficients, reference_y_range,
                                   bev_y_domain, H_inv,
                                   sample_num=LANE_DUAL_MOTION_SAMPLE_NUM):
    """在两条曲线共同可见范围内计算模型横坐标运动。"""
    if (current_coefficients is None or current_y_range is None or
            reference_coefficients is None or reference_y_range is None):
        return float("inf"), 0.0, 0.0

    current_start, current_end = sorted(map(float, current_y_range))
    reference_start, reference_end = sorted(map(float, reference_y_range))
    overlap_start = max(current_start, reference_start)
    overlap_end = min(current_end, reference_end)
    if overlap_end - overlap_start < 10.0:
        return float("inf"), 0.0, 0.0

    sample_y = np.linspace(overlap_start, overlap_end, max(4, int(sample_num)))
    current_model, current_valid = evaluate_curve_in_model(
        current_coefficients, sample_y, bev_y_domain, H_inv
    )
    reference_model, reference_valid = evaluate_curve_in_model(
        reference_coefficients, sample_y, bev_y_domain, H_inv
    )
    valid = current_valid & reference_valid
    if np.count_nonzero(valid) < 4:
        return float("inf"), 0.0, 0.0

    delta_x = current_model[valid, 0] - reference_model[valid, 0]
    motion_rms = float(np.sqrt(np.mean(delta_x ** 2)))
    median_dx = float(np.median(delta_x))
    if abs(median_dx) < 1e-6:
        sign_ratio = 0.0
    else:
        sign_ratio = float(np.mean(delta_x * np.sign(median_dx) > 0.0))
    return motion_rms, median_dx, sign_ratio


def calculate_aligned_curve_shape_error(coefficients_a, y_range_a,
                                        coefficients_b, y_range_b,
                                        bev_y_domain, H_inv):
    """消除整体横向平移后，比较两个大变化候选的形状差异。"""
    start_a, end_a = sorted(map(float, y_range_a))
    start_b, end_b = sorted(map(float, y_range_b))
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    if overlap_end - overlap_start < 20.0:
        return float("inf")

    sample_y = np.linspace(overlap_start, overlap_end, LANE_DUAL_MOTION_SAMPLE_NUM)
    model_a, valid_a = evaluate_curve_in_model(
        coefficients_a, sample_y, bev_y_domain, H_inv
    )
    model_b, valid_b = evaluate_curve_in_model(
        coefficients_b, sample_y, bev_y_domain, H_inv
    )
    valid = valid_a & valid_b
    if np.count_nonzero(valid) < 6:
        return float("inf")
    delta_x = model_a[valid, 0] - model_b[valid, 0]
    delta_x -= np.median(delta_x)
    return float(np.sqrt(np.mean(delta_x ** 2)))


def detect_dual_lane_common_motion(measurements, filtered_lanes,
                                   topology_valid, bev_y_domain, H_inv):
    """
    比较相邻两帧完整测量，识别左内/右内两条线共同发生的真实大弯。

    这里不能用跟踪输出作为参考：一旦某条线处于pending，两条跟踪器会
    不同步。上一帧高质量的完整测量更能表示真实帧间运动。
    """
    force_accept = np.zeros(XMEDIA_SVP_LANE_MAX_NUM, dtype=bool)
    left_idx, right_idx = 1, 2
    if not (topology_valid[left_idx] and topology_valid[right_idx]):
        return force_accept
    if not (PREV_MEASUREMENT_VALID[left_idx] and
            PREV_MEASUREMENT_VALID[right_idx]):
        return force_accept

    for lane_idx in (left_idx, right_idx):
        if measurements[lane_idx][1] is None:
            return force_accept
        if len(filtered_lanes[lane_idx]) < LANE_DUAL_MOTION_MIN_POINT_NUM:
            return force_accept
        if (LANE_FIT_MODEL_RMS[lane_idx] > LANE_DUAL_MOTION_MAX_FIT_RMS or
                LANE_FIT_MODEL_P90[lane_idx] > LANE_DUAL_MOTION_MAX_FIT_P90):
            return force_accept

    ranges = [
        measurements[left_idx][2], measurements[right_idx][2],
        PREV_MEASUREMENT_Y_RANGE[left_idx], PREV_MEASUREMENT_Y_RANGE[right_idx],
    ]
    overlap_start = max(min(map(float, y_range)) for y_range in ranges)
    overlap_end = min(max(map(float, y_range)) for y_range in ranges)
    if overlap_end - overlap_start < LANE_DUAL_MOTION_MIN_BEV_OVERLAP:
        return force_accept

    sample_y = np.linspace(
        overlap_start, overlap_end, LANE_DUAL_MOTION_SAMPLE_NUM
    )
    delta_x = []
    common_valid = np.ones(LANE_DUAL_MOTION_SAMPLE_NUM, dtype=bool)
    for lane_idx in (left_idx, right_idx):
        current_model, current_valid = evaluate_curve_in_model(
            measurements[lane_idx][1], sample_y, bev_y_domain, H_inv
        )
        previous_model, previous_valid = evaluate_curve_in_model(
            PREV_MEASUREMENT_COEF[lane_idx], sample_y, bev_y_domain, H_inv
        )
        common_valid &= current_valid & previous_valid
        delta_x.append(current_model[:, 0] - previous_model[:, 0])

    if np.count_nonzero(common_valid) < 10:
        return force_accept
    left_dx = delta_x[0][common_valid]
    right_dx = delta_x[1][common_valid]
    left_median = float(np.median(left_dx))
    right_median = float(np.median(right_dx))
    if left_median * right_median <= 0.0:
        return force_accept

    left_abs = abs(left_median)
    right_abs = abs(right_median)
    if min(left_abs, right_abs) < LANE_DUAL_MOTION_MIN_MEDIAN_DX:
        return force_accept
    magnitude_ratio = min(left_abs, right_abs) / max(left_abs, right_abs, 1e-6)
    if magnitude_ratio < LANE_DUAL_MOTION_MIN_MAG_RATIO:
        return force_accept

    left_sign_ratio = float(np.mean(left_dx * np.sign(left_median) > 0.0))
    right_sign_ratio = float(np.mean(right_dx * np.sign(right_median) > 0.0))
    pair_sign_ratio = float(np.mean(left_dx * right_dx > 0.0))
    pair_diff_rms = float(np.sqrt(np.mean((left_dx - right_dx) ** 2)))
    if (left_sign_ratio < LANE_DUAL_MOTION_MIN_LANE_SIGN_RATIO or
            right_sign_ratio < LANE_DUAL_MOTION_MIN_LANE_SIGN_RATIO or
            pair_sign_ratio < LANE_DUAL_MOTION_MIN_PAIR_SIGN_RATIO or
            pair_diff_rms > LANE_DUAL_MOTION_MAX_PAIR_DIFF_RMS):
        return force_accept

    force_accept[left_idx] = True
    force_accept[right_idx] = True
    LANE_COMMON_MOTION_ACCEPTED[left_idx] = 1
    LANE_COMMON_MOTION_ACCEPTED[right_idx] = 1
    return force_accept


def update_previous_measurement_cache(measurements, filtered_lanes,
                                      topology_valid):
    """保存本帧高质量完整测量，供下一帧共同运动识别。"""
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        curve_type, coefficients, y_range, confidence = measurements[lane_idx]
        measurement_reliable = (
            topology_valid[lane_idx] and
            coefficients is not None and
            y_range is not None and
            len(filtered_lanes[lane_idx]) >= LANE_DUAL_MOTION_MIN_POINT_NUM and
            LANE_FIT_MODEL_RMS[lane_idx] <= LANE_DUAL_MOTION_MAX_FIT_RMS and
            LANE_FIT_MODEL_P90[lane_idx] <= LANE_DUAL_MOTION_MAX_FIT_P90
        )
        if not measurement_reliable:
            PREV_MEASUREMENT_VALID[lane_idx] = 0
            continue
        PREV_MEASUREMENT_VALID[lane_idx] = 1
        PREV_MEASUREMENT_COEF[lane_idx] = coefficients
        PREV_MEASUREMENT_Y_RANGE[lane_idx] = y_range
        PREV_MEASUREMENT_POINT_NUM[lane_idx] = len(filtered_lanes[lane_idx])


def validate_lane_measurement_topology(measurements, filtered_lanes,
                                       bev_y_domain, H_inv):
    """
    用固定车道左右顺序抑制明显交叉的整段错分支。

    车道宽度会随变道、分合流和固定单应矩阵误差发生明显变化，因此这里
    不再强制宽度稳定；只有相邻两条线在近端已经反序，并且能明确判断哪
    一条偏离历史更大时才拒绝。
    返回的布尔数组只影响本帧测量；被拒绝的车道仍可由跟踪器短时预测。
    """
    valid_measurement = np.array(
        [measurement[1] is not None for measurement in measurements], dtype=bool
    )

    for left_idx in range(XMEDIA_SVP_LANE_MAX_NUM - 1):
        right_idx = left_idx + 1
        if not (valid_measurement[left_idx] and valid_measurement[right_idx]):
            continue
        if not (TRACKER_INITIALIZED[left_idx] and TRACKER_INITIALIZED[right_idx]):
            continue

        left_range = measurements[left_idx][2]
        right_range = measurements[right_idx][2]
        overlap_start = max(float(left_range[0]), float(right_range[0]))
        overlap_end = min(float(left_range[1]), float(right_range[1]))
        if overlap_end - overlap_start < 20.0:
            continue

        # 近端的车道间距更稳定，也不容易受单应变换地平线放大影响。
        sample_y = np.linspace(
            overlap_start + 0.65 * (overlap_end - overlap_start),
            overlap_end,
            6,
        )
        current_left, valid_cl = evaluate_curve_in_model(
            measurements[left_idx][1], sample_y, bev_y_domain, H_inv
        )
        current_right, valid_cr = evaluate_curve_in_model(
            measurements[right_idx][1], sample_y, bev_y_domain, H_inv
        )
        valid = valid_cl & valid_cr
        if np.count_nonzero(valid) < 3:
            continue

        current_width = current_right[valid, 0] - current_left[valid, 0]
        order_invalid = float(np.median(current_width)) <= 1.0
        if not order_invalid:
            continue

        left_error, _ = calculate_curve_fit_metrics(
            TRACKER_STATE[left_idx, :, 0], filtered_lanes[left_idx],
            bev_y_domain, H_inv
        )
        right_error, _ = calculate_curve_fit_metrics(
            TRACKER_STATE[right_idx, :, 0], filtered_lanes[right_idx],
            bev_y_domain, H_inv
        )
        if not (np.isfinite(left_error) and np.isfinite(right_error)):
            continue
        if abs(left_error - right_error) < LANE_ORDER_REJECT_ERROR_MARGIN:
            continue

        rejected_idx = left_idx if left_error > right_error else right_idx
        valid_measurement[rejected_idx] = False
        LANE_TOPOLOGY_REJECTED[rejected_idx] = 1

    return valid_measurement


def fit_smoothing_bspline(points, bev_y_domain,
                          smooth_lambda=LANE_SPLINE_SMOOTH_LAMBDA,
                          robust_iterations=4):
    """
    使用置信度、Huber权重和二阶差分约束拟合B样条。

    优化目标可直观理解为：
        数据误差（高置信点权重大） + lambda * 曲线弯折惩罚。
    Huber迭代继续降低大残差点的权重，二阶差分项则抑制8个控制量之间
    无意义的锯齿。这样比高阶多项式更具局部性，也更适合复合弯道。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < LANE_FIT_MIN_POINT_NUM:
        return None, None

    basis = build_bspline_basis(normalize_bev_y(points[:, 1], bev_y_domain))
    x = points[:, 0]
    base_weight = np.clip(points[:, 2], 0.05, 1.0)
    difference = np.zeros((LANE_SPLINE_COEF_NUM - 2,
                           LANE_SPLINE_COEF_NUM), dtype=np.float64)
    for idx in range(len(difference)):
        difference[idx, idx:idx + 3] = 1.0, -2.0, 1.0

    weight = base_weight.copy()
    coefficients = None
    for _ in range(max(1, robust_iterations)):
        normal = basis.T @ (weight[:, None] * basis)
        normal += smooth_lambda * (difference.T @ difference)
        normal += np.eye(LANE_SPLINE_COEF_NUM) * 1e-7
        rhs = basis.T @ (weight * x)
        try:
            coefficients = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return None, None

        residual = x - basis @ coefficients
        median = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - median))
        if sigma < 1e-3:
            break
        huber_limit = 2.5 * sigma
        robust_weight = np.ones(len(residual), dtype=np.float64)
        large = np.abs(residual) > huber_limit
        robust_weight[large] = huber_limit / np.abs(residual[large])
        weight = base_weight * robust_weight

    residual = x - basis @ coefficients
    return coefficients.astype(np.float32), residual.astype(np.float32)


def line_to_bspline_coefficients(line, bev_y_domain):
    """将 x=a*y+b 转换为统一的B样条系数。"""
    sample_y = np.linspace(bev_y_domain[0], bev_y_domain[1], 32)
    sample_x = line[0] * sample_y + line[1]
    basis = build_bspline_basis(normalize_bev_y(sample_y, bev_y_domain))
    coefficients, _, _, _ = np.linalg.lstsq(basis, sample_x, rcond=None)
    return coefficients.astype(np.float32)


def classify_and_fit_lane(points, bev_y_domain, H_inv):
    """
    同时评估直线与平滑B样条，在模型坐标中决定是否真的可以画直线。

    旧逻辑只看BEV残差，地平线附近很小的BEV误差映射回原图后可能已经
    很明显。现在只有在640x360坐标中也足够直时才使用严格直线，否则
    使用样条；连样条都无法解释的点集直接判无效，交给时序预测处理。
    """
    if len(points) < LANE_FIT_MIN_POINT_NUM:
        return LANE_CURVE_INVALID, None, None, 0.0

    line, line_residual = weighted_line_fit(points)
    if line is None:
        return LANE_CURVE_INVALID, None, None, 0.0

    y_range = np.array([np.min(points[:, 1]), np.max(points[:, 1])], dtype=np.float32)
    confidence = float(np.mean(points[:, 2]))

    line_coefficients = line_to_bspline_coefficients(line, bev_y_domain)
    line_model_rms, line_model_p90 = calculate_curve_fit_metrics(
        line_coefficients, points, bev_y_domain, H_inv
    )
    if (line_model_rms <= LANE_STRAIGHT_MAX_MODEL_RMS and
            line_model_p90 <= LANE_STRAIGHT_MAX_MODEL_P90):
        return LANE_CURVE_STRAIGHT, line_coefficients, y_range, confidence

    coefficients, spline_residual = fit_smoothing_bspline(points, bev_y_domain)
    if coefficients is None or not np.all(np.isfinite(coefficients)):
        return LANE_CURVE_INVALID, None, None, 0.0
    spline_bev_rms = float(np.sqrt(np.average(
        spline_residual ** 2, weights=np.clip(points[:, 2], 0.05, 1.0)
    )))
    spline_model_rms, spline_model_p90 = calculate_curve_fit_metrics(
        coefficients, points, bev_y_domain, H_inv
    )
    if (spline_bev_rms > LANE_FIT_MAX_BEV_RMS or
            spline_model_rms > LANE_FIT_MAX_MODEL_RMS or
            spline_model_p90 > LANE_FIT_MAX_MODEL_P90):
        return LANE_CURVE_INVALID, None, None, 0.0
    return LANE_CURVE_SPLINE, coefficients, y_range, confidence


def clear_lane_pending_candidate(lane_idx):
    """清空一条车道尚未确认的大变化候选。"""
    TRACKER_PENDING_COEF[lane_idx].fill(0)
    TRACKER_PENDING_Y_RANGE[lane_idx].fill(0)
    TRACKER_PENDING_CONFIDENCE[lane_idx] = 0
    TRACKER_PENDING_CURVE_TYPE[lane_idx] = LANE_CURVE_INVALID
    TRACKER_PENDING_COUNT[lane_idx] = 0
    TRACKER_PENDING_DIRECTION[lane_idx] = 0.0


def save_lane_pending_candidate(lane_idx, coefficients, y_range,
                                confidence, curve_type, direction,
                                pending_count=1):
    """保存本帧大变化，等待下一帧确认。"""
    TRACKER_PENDING_COEF[lane_idx] = coefficients
    TRACKER_PENDING_Y_RANGE[lane_idx] = y_range
    TRACKER_PENDING_CONFIDENCE[lane_idx] = confidence
    TRACKER_PENDING_CURVE_TYPE[lane_idx] = curve_type
    TRACKER_PENDING_COUNT[lane_idx] = max(1, int(pending_count))
    TRACKER_PENDING_DIRECTION[lane_idx] = float(direction)


def reset_single_lane_tracker(lane_idx):
    """清空指定车道的跟踪状态。"""
    TRACKER_INITIALIZED[lane_idx] = 0
    TRACKER_STATE[lane_idx].fill(0)
    TRACKER_P[lane_idx].fill(0)
    TRACKER_MISS[lane_idx] = 0
    TRACKER_Y_RANGE[lane_idx].fill(0)
    TRACKER_CONFIDENCE[lane_idx] = 0
    TRACKER_CURVE_TYPE[lane_idx] = LANE_CURVE_INVALID
    TRACKER_TYPE_SWITCH_COUNT[lane_idx] = 0
    clear_lane_pending_candidate(lane_idx)
    LANE_TRACK_STATUS[lane_idx] = LANE_TRACK_STATUS_INVALID


def initialize_lane_tracker(lane_idx, coefficients, y_range, confidence, curve_type):
    """用可靠拟合初始化一条车道的8个卡尔曼状态。"""
    TRACKER_INITIALIZED[lane_idx] = 1
    TRACKER_STATE[lane_idx, :, 0] = coefficients
    TRACKER_STATE[lane_idx, :, 1] = 0
    TRACKER_P[lane_idx].fill(0)
    TRACKER_P[lane_idx, :, 0, 0] = 25.0
    TRACKER_P[lane_idx, :, 1, 1] = 4.0
    TRACKER_MISS[lane_idx] = 0
    TRACKER_Y_RANGE[lane_idx] = y_range
    TRACKER_CONFIDENCE[lane_idx] = confidence
    TRACKER_CURVE_TYPE[lane_idx] = curve_type
    TRACKER_TYPE_SWITCH_COUNT[lane_idx] = 0
    clear_lane_pending_candidate(lane_idx)
    LANE_TRACK_STATUS[lane_idx] = LANE_TRACK_STATUS_MEASURED


def predict_lane_tracker(lane_idx):
    """对指定车道的所有B样条系数执行一次带速度衰减的预测。"""
    # 缺测时允许短时沿运动方向预测，但持续衰减，避免异常帧后越飞越远。
    transition = np.array([[1.0, 1.0], [0.0, 0.70]], dtype=np.float32)
    process_noise = np.array([[2.0, 0.0], [0.0, 0.50]], dtype=np.float32)
    for coef_idx in range(LANE_SPLINE_COEF_NUM):
        TRACKER_STATE[lane_idx, coef_idx] = transition @ TRACKER_STATE[lane_idx, coef_idx]
        covariance = TRACKER_P[lane_idx, coef_idx]
        TRACKER_P[lane_idx, coef_idx] = transition @ covariance @ transition.T + process_noise


def update_tracked_curve_type(lane_idx, measured_type):
    """连续两帧类型改变才切换，避免直线/曲线状态闪烁。"""
    if measured_type == TRACKER_CURVE_TYPE[lane_idx]:
        TRACKER_TYPE_SWITCH_COUNT[lane_idx] = 0
        return
    TRACKER_TYPE_SWITCH_COUNT[lane_idx] += 1
    if TRACKER_TYPE_SWITCH_COUNT[lane_idx] >= 2:
        TRACKER_CURVE_TYPE[lane_idx] = measured_type
        TRACKER_TYPE_SWITCH_COUNT[lane_idx] = 0


def track_lane_spline(lane_idx, measured_coefficients, measured_y_range,
                      measured_confidence, measured_type, measured_points,
                      bev_y_domain, H_inv, force_accept=False):
    """
    跟踪一条车道的B样条系数。
    测量无效时最多使用3帧预测，之后删除该跟踪。
    小变化使用卡尔曼平滑；可靠的中等变化立即跟随；双内线共同运动代表
    真实大弯并立即接受；其余单线大变化用连续候选确认。

    返回的是“当前应该画的轨迹”，不一定等于当前帧测量：
      - MEASURED：当前测量被接受，输出已更新；
      - PREDICTED：当前无可靠测量，最多沿旧速度预测3帧；
      - PENDING：看到单条线的大跳变，先画旧轨迹并等后续帧确认；
      - INVALID：没有可安全输出的轨迹。
    """
    measurement_valid = (
        measured_coefficients is not None and
        measured_y_range is not None and
        np.all(np.isfinite(measured_coefficients))
    )
    # 当前帧点数足、拟合误差小时，说明“模型点 -> 曲线”这一步
    # 是可靠的。这类测量不能因为和旧轨迹相差大就被画成旧线。
    measurement_high_quality = (
        measurement_valid and
        measured_points is not None and
        len(measured_points) >= LANE_DUAL_MOTION_MIN_POINT_NUM and
        LANE_FIT_MODEL_RMS[lane_idx] <= LANE_DUAL_MOTION_MAX_FIT_RMS and
        LANE_FIT_MODEL_P90[lane_idx] <= LANE_DUAL_MOTION_MAX_FIT_P90
    )

    if not TRACKER_INITIALIZED[lane_idx]:
        if not measurement_valid:
            return None, None, LANE_CURVE_INVALID, 0.0
        initialize_lane_tracker(
            lane_idx, measured_coefficients, measured_y_range,
            measured_confidence, measured_type
        )
        LANE_TRACK_MODEL_RMS[lane_idx] = 0.0
    else:
        predict_lane_tracker(lane_idx)
        predicted_error = float("inf")
        if measurement_valid and measured_points is not None and len(measured_points) > 0:
            predicted_error, _ = calculate_curve_fit_metrics(
                TRACKER_STATE[lane_idx, :, 0], measured_points,
                bev_y_domain, H_inv
            )
        if np.isfinite(predicted_error):
            LANE_TRACK_MODEL_RMS[lane_idx] = predicted_error

        if measurement_valid and force_accept:
            initialize_lane_tracker(
                lane_idx, measured_coefficients, measured_y_range,
                measured_confidence, measured_type
            )
            return (
                TRACKER_STATE[lane_idx, :, 0].copy(),
                TRACKER_Y_RANGE[lane_idx].copy(),
                int(TRACKER_CURVE_TYPE[lane_idx]),
                float(TRACKER_CONFIDENCE[lane_idx]),
            )

        # 两种测量可以直接进入更新：
        # 1) 自身点数和拟合质量非常好，即使离旧轨迹较远也应相信新观测；
        # 2) 与预测轨迹误差在接受门限内，属于正常帧间运动。
        measurement_accepted = (
            measurement_high_quality or
            (measurement_valid and
             np.isfinite(predicted_error) and
             predicted_error <= LANE_TRACK_ACCEPT_MODEL_RMS)
        )

        if measurement_accepted:
            clear_lane_pending_candidate(lane_idx)

            # 几何是否快速更新只由运动大小决定，直线/曲线类型仅作标签。
            if predicted_error > LANE_TRACK_FAST_UPDATE_MODEL_RMS:
                initialize_lane_tracker(
                    lane_idx, measured_coefficients, measured_y_range,
                    measured_confidence, measured_type
                )
                return (
                    TRACKER_STATE[lane_idx, :, 0].copy(),
                    TRACKER_Y_RANGE[lane_idx].copy(),
                    int(TRACKER_CURVE_TYPE[lane_idx]),
                    float(TRACKER_CONFIDENCE[lane_idx]),
                )

            # 稳定阶段才对8个样条系数分别做“位置+速度”卡尔曼平滑。
            # 大变化已在上方直接重置，否则强行平滑会令弯道/变道明显拖尾。
            measurement_noise = 4.0 / max(float(measured_confidence), 0.10)
            observation = np.array([1.0, 0.0], dtype=np.float32)
            identity = np.eye(2, dtype=np.float32)
            for coef_idx in range(LANE_SPLINE_COEF_NUM):
                state = TRACKER_STATE[lane_idx, coef_idx]
                covariance = TRACKER_P[lane_idx, coef_idx]
                innovation = measured_coefficients[coef_idx] - state[0]
                innovation_covariance = covariance[0, 0] + measurement_noise
                gain = covariance[:, 0] / innovation_covariance
                TRACKER_STATE[lane_idx, coef_idx] = state + gain * innovation
                TRACKER_P[lane_idx, coef_idx] = (identity - np.outer(gain, observation)) @ covariance

            TRACKER_MISS[lane_idx] = 0
            # 可见范围使用当前帧；所有额外补线只能由显式extension逻辑产生。
            TRACKER_Y_RANGE[lane_idx] = measured_y_range
            TRACKER_CONFIDENCE[lane_idx] = (
                0.70 * TRACKER_CONFIDENCE[lane_idx] + 0.30 * measured_confidence
            )
            update_tracked_curve_type(lane_idx, measured_type)
            LANE_TRACK_STATUS[lane_idx] = LANE_TRACK_STATUS_MEASURED
        else:
            if measurement_valid:
                _, motion_direction, _ = calculate_curve_motion_metrics(
                    measured_coefficients, measured_y_range,
                    TRACKER_STATE[lane_idx, :, 0], TRACKER_Y_RANGE[lane_idx],
                    bev_y_domain, H_inv
                )
                pending_matches = False
                if TRACKER_PENDING_COUNT[lane_idx] > 0:
                    shape_error = calculate_aligned_curve_shape_error(
                        measured_coefficients, measured_y_range,
                        TRACKER_PENDING_COEF[lane_idx],
                        TRACKER_PENDING_Y_RANGE[lane_idx],
                        bev_y_domain, H_inv
                    )
                    same_direction = (
                        abs(motion_direction) >= LANE_TRACK_PENDING_MIN_DIRECTION and
                        abs(float(TRACKER_PENDING_DIRECTION[lane_idx])) >=
                        LANE_TRACK_PENDING_MIN_DIRECTION and
                        motion_direction * TRACKER_PENDING_DIRECTION[lane_idx] > 0.0
                    )
                    pending_matches = (
                        same_direction or
                        (np.isfinite(shape_error) and
                         shape_error <= LANE_TRACK_PENDING_SHAPE_RMS)
                    )

                if pending_matches:
                    # 连续两帧都是高质量大变化且方向/形状一致，切换到当前帧。
                    initialize_lane_tracker(
                        lane_idx, measured_coefficients, measured_y_range,
                        measured_confidence, measured_type
                    )
                    return (
                        TRACKER_STATE[lane_idx, :, 0].copy(),
                        TRACKER_Y_RANGE[lane_idx].copy(),
                        int(TRACKER_CURVE_TYPE[lane_idx]),
                        float(TRACKER_CONFIDENCE[lane_idx]),
                    )

                pending_age = min(
                    int(TRACKER_PENDING_COUNT[lane_idx]) + 1,
                    LANE_TRACK_PENDING_MAX_AGE,
                )
                if pending_age >= LANE_TRACK_PENDING_MAX_AGE:
                    # 连续多帧都有有效新测量时，不能无限期画旧轨迹。
                    initialize_lane_tracker(
                        lane_idx, measured_coefficients, measured_y_range,
                        measured_confidence, measured_type
                    )
                    return (
                        TRACKER_STATE[lane_idx, :, 0].copy(),
                        TRACKER_Y_RANGE[lane_idx].copy(),
                        int(TRACKER_CURVE_TYPE[lane_idx]),
                        float(TRACKER_CONFIDENCE[lane_idx]),
                    )

                save_lane_pending_candidate(
                    lane_idx, measured_coefficients, measured_y_range,
                    measured_confidence, measured_type, motion_direction,
                    pending_count=pending_age,
                )
                LANE_TRACK_STATUS[lane_idx] = LANE_TRACK_STATUS_PENDING
                # 能进入pending的已经不是严格高质量测量。它可以作为
                # 候选，但不能把旧轨迹的丢失计数清零，否则零散少点
                # 会让旧线无限期残留。
                TRACKER_MISS[lane_idx] += 1
                TRACKER_CONFIDENCE[lane_idx] *= 0.90
                if TRACKER_MISS[lane_idx] > LANE_TRACK_MAX_MISS:
                    reset_single_lane_tracker(lane_idx)
                    return None, None, LANE_CURVE_INVALID, 0.0
            else:
                # 质量不合格不是候选变化，必须连续两帧重新确认。
                clear_lane_pending_candidate(lane_idx)
                LANE_TRACK_STATUS[lane_idx] = LANE_TRACK_STATUS_PREDICTED
                TRACKER_MISS[lane_idx] += 1
                TRACKER_CONFIDENCE[lane_idx] *= 0.90
                if TRACKER_MISS[lane_idx] > LANE_TRACK_MAX_MISS:
                    reset_single_lane_tracker(lane_idx)
                    return None, None, LANE_CURVE_INVALID, 0.0

    tracked_coefficients = TRACKER_STATE[lane_idx, :, 0].copy()

    return (
        tracked_coefficients,
        TRACKER_Y_RANGE[lane_idx].copy(),
        int(TRACKER_CURVE_TYPE[lane_idx]),
        float(TRACKER_CONFIDENCE[lane_idx]),
    )


def calculate_fitted_model_length(coefficients, y_range, bev_y_domain, H_inv):
    """在延长前，计算拟合线逆变换到640x360后的像素长度。"""
    if coefficients is None or y_range is None:
        return 0.0
    y_start, y_end = sorted((float(y_range[0]), float(y_range[1])))
    if y_end - y_start < 1e-3:
        return 0.0
    sample_y = np.linspace(y_start, y_end, 128)
    sample_x = evaluate_bspline(coefficients, sample_y, bev_y_domain)
    bev_points = np.column_stack((sample_x, sample_y))
    image_points, valid = perspective_transform_points(bev_points, H_inv)
    image_points = image_points[valid]
    if len(image_points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(image_points, axis=0), axis=1)))


def calculate_observed_model_geometry(bev_points, H_inv, model_height):
    """返回当前实测点在640x360中的远端、近端和纵向覆盖。"""
    if bev_points is None or len(bev_points) == 0:
        return None
    model_points, valid = perspective_transform_points(
        np.asarray(bev_points)[:, :2], H_inv
    )
    model_y = model_points[valid, 1]
    model_y = model_y[np.isfinite(model_y)]
    if len(model_y) < 2:
        return None

    height = max(float(model_height), 1e-6)
    far_y = float(np.min(model_y))
    near_y = float(np.max(model_y))
    span = max(0.0, near_y - far_y)
    return {
        "far_y": far_y,
        "near_y": near_y,
        "span": span,
        "far_ratio": far_y / height,
        "near_ratio": near_y / height,
        "span_ratio": span / height,
    }


def calculate_chord_deviation(model_points):
    """计算曲线点到首尾弦线的最大距离，单位为模型像素。"""
    model_points = np.asarray(model_points, dtype=np.float64)
    if len(model_points) < 3:
        return 0.0
    chord = model_points[-1] - model_points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-6:
        return float("inf")
    relative = model_points - model_points[0]
    distance = np.abs(
        chord[0] * relative[:, 1] - chord[1] * relative[:, 0]
    ) / chord_length
    return float(np.max(distance))


def calculate_curve_bend_metrics(coefficients, y_range, bev_y_domain, H_inv):
    """
    在模型坐标中评估整段弯曲和远端弯曲。

    延长发生在远端，因此远端指标用来决定切线外推的上限；
    整段指标则防止大弯道被误当成短直线大幅度延长。
    """
    if coefficients is None or y_range is None:
        return float("inf"), float("inf"), float("inf")
    y_start, y_end = sorted(map(float, y_range))
    if y_end - y_start < 1e-3:
        return float("inf"), float("inf"), float("inf")

    sample_y = np.linspace(y_start, y_end, 64)
    model_points, valid = evaluate_curve_in_model(
        coefficients, sample_y, bev_y_domain, H_inv
    )
    model_points = model_points[valid]
    if len(model_points) < 12:
        return float("inf"), float("inf"), float("inf")

    full_deviation = calculate_chord_deviation(model_points)
    far_num = max(8, int(math.ceil(0.35 * len(model_points))))
    far_points = model_points[:far_num]
    far_deviation = calculate_chord_deviation(far_points)

    segment_num = max(3, far_num // 3)
    first_vector = far_points[segment_num] - far_points[0]
    last_vector = far_points[-1] - far_points[-1 - segment_num]
    first_angle = math.atan2(first_vector[1], first_vector[0])
    last_angle = math.atan2(last_vector[1], last_vector[0])
    angle_difference = math.atan2(
        math.sin(last_angle - first_angle), math.cos(last_angle - first_angle)
    )
    far_heading_change = abs(math.degrees(angle_difference))
    return full_deviation, far_deviation, far_heading_change


def get_extension_length_by_vertical_coverage(observed_geometry,
                                              bend_metrics,
                                              bev_y_domain,
                                              valid_point_num):
    """
    根据实测点的纵向覆盖判断是否需要向远端补线。

    只有近端仍可见、远端确实变短的测量才会延长。直线可以按档位
    补齐；远端已明显弯曲时只允许小幅切线外推，避免弯道被拉直。

    注意：模型图中“远端”是较小的图像y；在本BEV定义下对应从观测
    y_start继续向更小的BEV y补点。这里返回BEV长度，不是原图像素长度。
    """
    if (valid_point_num < LANE_EXTENSION_MIN_POINT_NUM or
            observed_geometry is None):
        return 0.0

    far_ratio = float(observed_geometry["far_ratio"])
    near_ratio = float(observed_geometry["near_ratio"])
    span_ratio = float(observed_geometry["span_ratio"])
    if (near_ratio < 0.85 or span_ratio < 0.20 or far_ratio > 0.78 or
            span_ratio >= 0.55 or far_ratio <= 0.44):
        return 0.0

    # 同时使用远端位置和可见高度，指标冲突时不冒进外推。
    extension_ratio = 0.0
    if 0.45 <= span_ratio < 0.55 and 0.44 < far_ratio <= 0.55:
        extension_ratio = 0.04
    elif 0.32 <= span_ratio < 0.45 and 0.52 <= far_ratio <= 0.66:
        extension_ratio = 0.07
    elif 0.20 <= span_ratio < 0.32 and 0.65 <= far_ratio <= 0.78:
        extension_ratio = 0.10
    if extension_ratio <= 0.0:
        return 0.0

    full_deviation, far_deviation, far_heading_change = bend_metrics
    if not np.all(np.isfinite(bend_metrics)) or far_deviation > 2.0:
        return 0.0
    if far_deviation > 0.80:
        extension_ratio = min(extension_ratio, 0.02)
    elif far_deviation > 0.35:
        extension_ratio = min(extension_ratio, 0.04)
    if full_deviation > 6.0 or far_heading_change > 8.0:
        extension_ratio = min(extension_ratio, 0.02)

    domain_length = max(float(bev_y_domain[1] - bev_y_domain[0]), 1e-6)
    return extension_ratio * domain_length


def reject_unstable_tangent_extension(coefficients, y_range, bev_y_domain,
                                      extension_length):
    """远端原始切线过斜时取消延长，避免裁剪斜率后产生折钩。"""
    if extension_length <= 0.0 or coefficients is None or y_range is None:
        return 0.0
    y_start, y_end = sorted(map(float, y_range))
    domain_length = max(float(bev_y_domain[1] - bev_y_domain[0]), 1e-6)
    probe_y = min(y_end, y_start + max(1.0, 0.005 * domain_length))
    if probe_y - y_start < 1e-6:
        return 0.0
    start_x = float(evaluate_bspline(coefficients, [y_start], bev_y_domain)[0])
    probe_x = float(evaluate_bspline(coefficients, [probe_y], bev_y_domain)[0])
    raw_tangent = (probe_x - start_x) / (probe_y - y_start)
    if not np.isfinite(raw_tangent) or abs(raw_tangent) > 3.0:
        return 0.0
    return float(extension_length)


def generate_fitted_bev_lane(lane_idx, coefficients, y_range, confidence,
                             bev_y_domain, extension_length=0.0):
    """
    在有效BEV y范围内生成最多360个稠密点。
    延长部分使用远端切线，避免B样条超出观测范围后突然甩弯。
    """
    if coefficients is None or y_range is None:
        return np.empty((0, 3), dtype=np.float32)
    y_start, y_end = sorted((float(y_range[0]), float(y_range[1])))
    if y_end - y_start < 1e-3:
        return np.empty((0, 3), dtype=np.float32)

    extended_start = max(
        float(bev_y_domain[0]), y_start - max(float(extension_length), 0.0)
    )
    requested_num = int(np.ceil(y_end - extended_start)) + 1
    point_num = min(XMEDIA_SVP_LANE_MAX_FITTED_POINT, max(2, requested_num))
    sample_y = np.linspace(extended_start, y_end, point_num, dtype=np.float32)
    sample_x = evaluate_bspline(coefficients, sample_y, bev_y_domain).astype(np.float32)

    extension_mask = sample_y < y_start
    if np.any(extension_mask):
        domain_length = max(float(bev_y_domain[1] - bev_y_domain[0]), 1e-6)
        # 用边界附近的小步长估计真实导数，避免跨过过长弯曲段
        # 得到弦线方向，在延长段与实测段交界处形成小折钩。
        probe_y = min(y_end, y_start + max(1.0, 0.005 * domain_length))
        start_x = float(evaluate_bspline(coefficients, [y_start], bev_y_domain)[0])
        probe_x = float(evaluate_bspline(coefficients, [probe_y], bev_y_domain)[0])
        tangent = (probe_x - start_x) / max(probe_y - y_start, 1e-6)
        tangent = float(np.clip(tangent, -3.0, 3.0))
        sample_x[extension_mask] = (
            start_x + tangent * (sample_y[extension_mask] - y_start)
        )
    points = np.column_stack((sample_x, sample_y,
                              np.full(point_num, confidence, dtype=np.float32)))
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]

    point_num = min(len(points), XMEDIA_SVP_LANE_MAX_FITTED_POINT)
    BEV_FITTED_RESULT[lane_idx, :point_num] = points[:point_num]
    BEV_FITTED_MASK[lane_idx, :point_num] = 1
    BEV_FITTED_POINT_NUM[lane_idx] = point_num
    return points[:point_num]


def find_stable_far_endpoint_index(fitted_model_points, observed_bev_points,
                                   H_inv):
    """
    返回需要裁掉的远端过冲点数。

    只在原始点已明确指向一侧，而拟合线在最远的10个model像素内
    先向反方向走、随后又恢复时才裁剪。这样只删除B样条端点的
    小折返，不会把真实大弯道拉直。
    """
    fitted_model_points = np.asarray(fitted_model_points, dtype=np.float64)
    if (observed_bev_points is None or
            len(observed_bev_points) < LANE_FAR_TRIM_MIN_POINT_NUM or
            len(fitted_model_points) < 6):
        return 0

    observed_model, observed_valid = perspective_transform_points(
        np.asarray(observed_bev_points)[:, :2], H_inv
    )
    observed_model = observed_model[observed_valid]
    observed_model = observed_model[np.all(np.isfinite(observed_model), axis=1)]
    if len(observed_model) < LANE_FAR_TRIM_MIN_POINT_NUM:
        return 0
    observed_model = observed_model[np.argsort(observed_model[:, 1])]

    raw_far = observed_model[:min(8, len(observed_model))]
    raw_delta = np.diff(raw_far, axis=0)
    valid_delta = raw_delta[:, 1] > 0.5
    if np.count_nonzero(valid_delta) < 4:
        return 0
    raw_slopes = raw_delta[valid_delta, 0] / raw_delta[valid_delta, 1]
    raw_slope = float(np.median(raw_slopes))
    if abs(raw_slope) < LANE_FAR_TRIM_MIN_RAW_SLOPE:
        return 0
    expected_direction = 1.0 if raw_slope > 0.0 else -1.0

    far_y = float(raw_far[0, 1])
    near_y = float(observed_model[-1, 1])
    observed_span = max(near_y - far_y, 1e-6)
    analysis_end = far_y + min(
        LANE_FAR_TRIM_ANALYSIS_SPAN, 0.08 * observed_span
    )
    analysis_indices = np.flatnonzero(
        (fitted_model_points[:, 1] >= far_y - 0.5) &
        (fitted_model_points[:, 1] <= analysis_end)
    )
    if len(analysis_indices) < 3:
        return 0

    analysis_x = fitted_model_points[analysis_indices, 0]
    if expected_direction < 0.0:
        local_extreme = int(np.argmax(analysis_x))
        raw_wrong_excursion = max(0.0, float(np.max(raw_far[:, 0]) - raw_far[0, 0]))
    else:
        local_extreme = int(np.argmin(analysis_x))
        raw_wrong_excursion = max(0.0, float(raw_far[0, 0] - np.min(raw_far[:, 0])))
    if local_extreme == 0 or local_extreme == len(analysis_indices) - 1:
        return 0

    first_idx = int(analysis_indices[0])
    extreme_idx = int(analysis_indices[local_extreme])
    fitted_wrong_excursion = (
        fitted_model_points[extreme_idx, 0] - fitted_model_points[first_idx, 0]
    ) * -expected_direction
    if (fitted_wrong_excursion - raw_wrong_excursion <
            LANE_FAR_TRIM_MIN_EXCURSION):
        return 0
    if fitted_model_points[extreme_idx, 1] - far_y > LANE_FAR_TRIM_MAX_LENGTH:
        return 0

    confirm_end = far_y + min(
        LANE_FAR_TRIM_CONFIRM_SPAN, 0.15 * observed_span
    )
    confirm_indices = np.flatnonzero(
        (fitted_model_points[:, 1] > fitted_model_points[extreme_idx, 1]) &
        (fitted_model_points[:, 1] <= confirm_end)
    )
    if len(confirm_indices) == 0:
        return 0
    recovered_distance = (
        fitted_model_points[int(confirm_indices[-1]), 0] -
        fitted_model_points[extreme_idx, 0]
    ) * expected_direction
    if recovered_distance < LANE_FAR_TRIM_MIN_RECOVERY:
        return 0
    return extreme_idx


def convert_fitted_lane_to_original(lane_idx, bev_points, H_inv,
                                    original_width, original_height,
                                    model_width, model_height,
                                    observed_bev_points=None):
    """将BEV稠密点逆变换到模型图，再映射到原图坐标。"""
    if len(bev_points) == 0:
        return
    image_points, valid = perspective_transform_points(bev_points[:, :2], H_inv)
    valid &= image_points[:, 0] >= 0
    valid &= image_points[:, 0] < model_width
    valid &= image_points[:, 1] >= 0
    valid &= image_points[:, 1] < model_height
    image_points = image_points[valid]
    scores = bev_points[valid, 2]
    if len(image_points) == 0:
        return

    stable_start = find_stable_far_endpoint_index(
        image_points, observed_bev_points, H_inv
    )
    LANE_FAR_TRIMMED_POINT_NUM[lane_idx] = stable_start
    if stable_start > 0:
        image_points = image_points[stable_start:]
        scores = scores[stable_start:]
    if len(image_points) == 0:
        return

    image_points[:, 0] *= float(original_width) / float(model_width)
    image_points[:, 1] *= float(original_height) / float(model_height)
    point_num = min(len(image_points), XMEDIA_SVP_LANE_MAX_FITTED_POINT)
    ROW_FITTED_RESULT[lane_idx, :point_num, 0:2] = image_points[:point_num]
    ROW_FITTED_RESULT[lane_idx, :point_num, 2] = scores[:point_num]
    ROW_FITTED_MASK[lane_idx, :point_num] = 1
    LANE_FITTED_POINT_NUM[lane_idx] = point_num

def get_lane_color(lane_idx):
    """固定四条车道（左外、左内、右内、右外）的BGR颜色。"""
    colors = ((255, 0, 0), (0, 140, 255), (0, 255, 255), (0, 255, 128))
    return colors[lane_idx]


def draw_raw_lane_points(vis, model_width=640, model_height=360):
    """可选调试：把模型原始点映射到原图后画成小圆点。"""
    scale_x = vis.shape[1] / float(model_width)
    scale_y = vis.shape[0] / float(model_height)
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        color = get_lane_color(lane_idx)
        for row_idx in np.flatnonzero(ROW_MASK[lane_idx]):
            x = int(round(ROW_RESULT[lane_idx, row_idx, 0] * scale_x))
            y = int(round(ROW_RESULT[lane_idx, row_idx, 1] * scale_y))
            cv2.circle(vis, (x, y), 3, color, -1)
    return vis


def draw_fitted_lanes(vis):
    """绘制拟合和跟踪后的最终原图坐标。"""
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        point_num = LANE_FITTED_POINT_NUM[lane_idx]
        if point_num < 2:
            continue
        points = ROW_FITTED_RESULT[lane_idx, :point_num, :2]
        points = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [points], False, get_lane_color(lane_idx), 5, cv2.LINE_AA)
    return vis


def draw_lane_on_image(vis, draw_raw=False, model_width=640, model_height=360):
    """在图像上绘制最终车道线和标签。"""
    if draw_raw:
        vis = draw_raw_lane_points(vis, model_width, model_height)
    vis = draw_fitted_lanes(vis)
    
    # step = 1200
    # for num, lane in enumerate(filter_coords):
    #     if len(lane) == 0:
    #         continue

    #     n_points = len(lane)  

    #     # --- 求 y_min, y_max ---
    #     y_min = lane[0][1]
    #     y_max = lane[0][1]
    #     for q in range(n_points):
    #         y_min = min(y_min, lane[q][1])
    #         y_max = max(y_max, lane[q][1])

    #     h_s = y_max - y_min

    #     # --- 沿 Y 轴等间距重采样 ---
    #     for k in range(step):
    #         tmp_y = (k / step) * h_s + y_min

    #         # --- 找到 tmp_y 所落的线段，线性插值求 tmp_x ---
    #         found = False
    #         for j in range(n_points - 1):
    #             y0 = lane[j][1]
    #             y1 = lane[j + 1][1]

    #             if y0 <= tmp_y <= y1:
    #                 x0 = lane[j][0]
    #                 x1 = lane[j + 1][0]

    #                 if abs(y1 - y0) > 1e-6:
    #                     t = (tmp_y - y0) / (y1 - y0)
    #                     tmp_x = x0 + t * (x1 - x0)
    #                 else:
    #                     tmp_x = (x0 + x1) / 2.0  # y 相同时取中点

    #                 found = True
    #                 break

    #         # --- 超出范围时用最近端点 ---
    #         if not found:
    #             if tmp_y < lane[0][1]:
    #                 tmp_x = lane[0][0]
    #             else:
    #                 tmp_x = lane[-1][0]

    #         x = int(round(tmp_x))
    #         y = int(round(tmp_y))

    #         cv2.circle(vis, (x, y), 4, (0, 0, 0), -1)

    # 绘制标签文字（使用PIL支持中文）
    for lane_index, lane_labels in enumerate(LANE_LABEL_RESULT):
        draw_point = (20, 40 + lane_index * 80)

        label = ''
        for l, label_index in enumerate(lane_labels):
            if label_index == 1:
                label += lane_label_index_dic[l] + ' '

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
        text = "车道线 {} 类型: {}".format(lane_index, label)
        draw.text(draw_point, text, font=font, fill=(255, 0, 0))
        vis = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
    return vis




def perspective_transform_points(points, matrix, min_denominator=1e-6):
    """
    对N个(x, y)点执行单应变换，同时返回分母有效标志。

    齐次坐标计算为 [u,v,w]^T = H[x,y,1]^T，结果是(u/w,v/w)。
    地平线附近w可能接近0，此时坐标会被放大到极端值，所以不能只判断
    NaN，还必须用min_denominator主动丢弃不稳定点。
    """
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=bool)

    homogeneous = np.ones((len(points), 3), dtype=np.float64)
    homogeneous[:, :2] = points[:, :2]
    transformed = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    denominator = transformed[:, 2]
    valid = np.isfinite(denominator) & (np.abs(denominator) >= min_denominator)

    result = np.zeros((len(points), 2), dtype=np.float32)
    result[valid] = (transformed[valid, :2] / denominator[valid, None]).astype(np.float32)
    valid &= np.all(np.isfinite(result), axis=1)
    return result, valid

def get_bev_y_domain(H, model_width):
    """
    根据固定H和所有row锚点生成跨帧一致的BEV y归一化范围。

    这里必须使用与原始点转换相同的分母阈值。旧逻辑把分母接近0、
    实际一定会被丢弃的地平线点也放进样条定义域，造成前一大段控制量
    永远没有观测，随后在跟踪或直线投影时被严重放大。
    """
    sample_points = []
    for y in ROW_COORDS:
        for x in (0.0, model_width * 0.5, float(model_width)):
            sample_points.append((x, float(y)))
    transformed, valid = perspective_transform_points(sample_points, H, min_denominator=HOMOGRAPHY_MIN_DENOMINATOR)
    valid_y = transformed[valid, 1]
    if len(valid_y) < 2:
        raise ValueError("无法从单应矩阵得到有效的BEV y范围")
    y_min = float(np.min(valid_y))
    y_max = float(np.max(valid_y))
    if y_max - y_min < 1e-3:
        raise ValueError("BEV y范围过小，无法进行车道线拟合")
    return y_min, y_max

def convert_raw_lanes_to_bev(H):
    """
    将ROW_RESULT中的640x360原始点转换到BEV_RESULT。

    source_index仍沿用58个row锚点的下标，便于逐阶段对照调试；第三列
    score原样传递，后续过滤和拟合会把它当作可信度权重。
    """
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        valid_indices = np.flatnonzero(ROW_MASK[lane_idx])
        if len(valid_indices) == 0:
            continue

        raw_points = ROW_RESULT[lane_idx, valid_indices, :2]
        bev_points, valid = perspective_transform_points(raw_points, H, min_denominator=HOMOGRAPHY_MIN_DENOMINATOR)
        for source_index, bev_point, point_valid in zip(valid_indices, bev_points, valid):
            if not point_valid:
                continue
            BEV_RESULT[lane_idx, source_index, 0:2] = bev_point
            BEV_RESULT[lane_idx, source_index, 2] = ROW_RESULT[lane_idx, source_index, 2]
            BEV_MASK[lane_idx, source_index] = 1
            BEV_POINT_NUM[lane_idx] += 1

def lane_post_optimize(H, original_width, original_height, model_width, model_height):
    """
    单帧后处理总入口（阅读后处理时先从这里开始）：
    原始点 -> BEV -> 飞点过滤 -> 直线/B样条 -> 跟踪 -> 稠密点 -> 原图。

    参数中的model_height应是裁剪映射前的完整模型坐标高度。本文件调用时
    传入train_height/crop_ratio，即288/0.8=360；ROW_COORDS也定义在
    这个640x360坐标系。original_width/height则是最终要画线的原图尺寸。

    一个极简心智例子：某条线有20个原始点，其中1个横向飞出很远。
    转BEV后局部过滤删掉该点；剩余19点若接近直线就得到STRAIGHT，否则
    得到8系数SPLINE。若下一帧漏检，跟踪器短时输出PREDICTED旧轨迹；
    恢复检测后再更新，最后采样为稠密点并变回原图坐标绘制。
    """
    # Step 1：准备正/逆坐标变换。
    # H负责模型透视图 -> BEV；H_inv负责BEV -> 模型透视图。
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    # 固定定义域使B样条8个系数在所有帧、所有可见长度下保持相同含义。
    bev_y_domain = get_bev_y_domain(H, model_width)

    # Step 2：只做坐标变换，不在这里直接拟合，保留BEV原始中间结果。
    convert_raw_lanes_to_bev(H)

    # Step 3（空间处理）：先生成四条车道的当前帧候选，暂不更新跟踪器。
    # 这样拓扑检查时四条线看到的都还是同一个“上一帧状态”，不会因
    # lane0已经更新、lane1尚未更新而引入依赖处理顺序的不一致。
    # measurement固定为(curve_type, coefficients, y_range, confidence)。
    filtered_lanes = []
    measurements = []
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        filtered_points = filter_bev_lane_points(lane_idx)
        save_filtered_bev_lane(lane_idx, filtered_points)
        filtered_lanes.append(filtered_points)

        measurement = classify_and_fit_lane(
            filtered_points, bev_y_domain, H_inv
        )
        measurements.append(measurement)
        curve_type, coefficients, y_range, confidence = measurement
        if coefficients is not None:
            LANE_CURVE_TYPE[lane_idx] = curve_type
            LANE_SPLINE_COEF[lane_idx] = coefficients
            LANE_SPLINE_Y_RANGE[lane_idx] = y_range
            fit_model_rms, fit_model_p90 = calculate_curve_fit_metrics(
                coefficients, filtered_points, bev_y_domain, H_inv
            )
            if np.isfinite(fit_model_rms):
                LANE_FIT_MODEL_RMS[lane_idx] = fit_model_rms
            if np.isfinite(fit_model_p90):
                LANE_FIT_MODEL_P90[lane_idx] = fit_model_p90

    # Step 4（多线关系）：检查相邻车道是否发生左右反序/交叉。
    # 这里只否决明显更可疑的那条“当前测量”，并非立刻删除它的旧轨迹。
    topology_valid = validate_lane_measurement_topology(
        measurements, filtered_lanes, bev_y_domain, H_inv
    )
    # 四条线同时没有任何可靠当前测量时，继续画上一场景
    # 的预测线比短暂不画更危险，特别是镜头切换或大幅横摆。
    if not np.any(topology_valid):
        reset_lane_trackers()
    # 若左右内线同时同向大幅运动，更像车辆/道路真实转弯，而不是两条线
    # 同时误检，因此允许它们绕过单线大跳变的pending等待，立即更新。
    force_accept = detect_dual_lane_common_motion(
        measurements, filtered_lanes, topology_valid, bev_y_domain, H_inv
    )

    # Step 5（时间处理）：逐线执行跟踪、短时预测或大变化确认。
    # Step 6（输出几何）：只对“当前可靠实测”考虑远端延长；预测线不补长。
    # Step 7（坐标输出）：生成BEV稠密点，逆变换并缩放到原图。
    for lane_idx in range(XMEDIA_SVP_LANE_MAX_NUM):
        curve_type, coefficients, y_range, confidence = measurements[lane_idx]
        filtered_points = filtered_lanes[lane_idx]
        if not topology_valid[lane_idx]:
            curve_type = LANE_CURVE_INVALID
            coefficients = None
            y_range = None
            confidence = 0.0

        tracked_coefficients, tracked_y_range, tracked_type, tracked_confidence = track_lane_spline(
            lane_idx,
            coefficients,
            y_range,
            confidence,
            curve_type,
            filtered_points,
            bev_y_domain,
            H_inv,
            force_accept=force_accept[lane_idx],
        )
        if tracked_coefficients is None:
            continue

        # 当前最终类型使用跟踪器的带迟滞状态。
        LANE_CURVE_TYPE[lane_idx] = tracked_type
        processed_length = calculate_fitted_model_length(
            tracked_coefficients, tracked_y_range, bev_y_domain, H_inv
        )
        # 预测/候选状态虽然可以短时继续画，但它不代表本帧真的观察到了
        # 对应范围，所以不能拿它触发延长，否则会把历史信息越画越远。
        measurement_is_current = (
            LANE_TRACK_STATUS[lane_idx] == LANE_TRACK_STATUS_MEASURED and
            topology_valid[lane_idx]
        )
        reliable_point_num = len(filtered_points) if measurement_is_current else 0
        observed_geometry = calculate_observed_model_geometry(
            filtered_points, H_inv, model_height
        ) if measurement_is_current else None
        bend_metrics = calculate_curve_bend_metrics(
            tracked_coefficients, tracked_y_range, bev_y_domain, H_inv
        )
        extension_length = get_extension_length_by_vertical_coverage(
            observed_geometry, bend_metrics, bev_y_domain, reliable_point_num
        )
        extension_length = reject_unstable_tangent_extension(
            tracked_coefficients, tracked_y_range, bev_y_domain,
            extension_length
        )
        LANE_PROCESSED_MODEL_LENGTH[lane_idx] = processed_length
        if observed_geometry is not None:
            LANE_OBSERVED_MODEL_SPAN[lane_idx] = observed_geometry["span"]
        if np.isfinite(bend_metrics[1]):
            LANE_CURVE_FAR_DEVIATION[lane_idx] = bend_metrics[1]
        LANE_EXTENSION_BEV_LENGTH[lane_idx] = extension_length
        bev_fitted_points = generate_fitted_bev_lane(
            lane_idx,
            tracked_coefficients,
            tracked_y_range,
            tracked_confidence,
            bev_y_domain,
            extension_length,
        )
        convert_fitted_lane_to_original(
            lane_idx,
            bev_fitted_points,
            H_inv,
            original_width,
            original_height,
            model_width,
            model_height,
            observed_bev_points=(
                filtered_points if measurement_is_current else None
            ),
        )

    # Step 8：把本帧高质量完整测量保存为“下一帧的上一帧”。
    # 必须放在共同运动判断之后，否则比较的会是当前帧和当前帧，运动量为0。
    update_previous_measurement_cache(
        measurements, filtered_lanes, topology_valid
    )

def run_video_mode(session, input_name, output_names, image_list, args):
    """视频模式：将带车道线的图片合成视频"""
    batch_size = 1
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
    if not vout.isOpened():
        print(f"Error: 无法创建视频 {video_path}")
        return
    print(f'[视频模式] 共 {len(image_list)} 帧, 保存到 {video_path}')

    # 加载变换矩阵
    data = [[-1.0404464645, -2.4681873580, 604.5821286670],
            [-0.0041529616, -7.1172920149, 898.0927908948],
            [-0.0000051994, -0.0083944402, 1.0000000000]]
  
    H = np.array(data, dtype=np.float32)
    reset_lane_trackers()

    for batch_start in tqdm(range(0, len(image_list), batch_size), total=math.ceil(len(image_list) / batch_size), desc='生成视频'):
        img_path = image_list[batch_start]
        img_transposed, h, w = preprocess_image(img_path, args.train_height, args.train_width, args.crop_ratio)
        if img_transposed is None:
            print(f"Warning: 跳过无法读取的图片 {img_path}")
            continue

        # CHW -> NCHW，即 (3, H, W) -> (1, 3, H, W)
        input_data = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        # 处理新帧前必须清零，否则会混入上一帧数据。
        reset_frame_results()
        pred = onnx_inference(session, input_name, output_names, input_data)
        pred2coords(pred, train_width=args.train_width)

        # 后处理内部按顺序完成BEV转换、拟合、跟踪和逆变换。
        lane_post_optimize(H, w, h, args.train_width, args.train_height/args.crop_ratio)

        vis = cv2.imread(img_path)
        if vis is None:
            continue
        vis = draw_lane_on_image(
            vis,
            draw_raw=True,
            model_width=args.train_width,
            model_height=args.train_height/args.crop_ratio,
        )
        if vis.shape[:2] != (img_h, img_w):
            vis = cv2.resize(vis, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        vout.write(vis)
    vout.release()
    print("✅ 视频模式处理完成!")

def main():
    args = parse_args()
    session, input_name, output_names = load_onnx_model(args.onnx_model)
    image_list = collect_images(data_root=args.demo_data_root)
    if len(image_list) == 0:
        print("未找到任何图片，退出。")
        return
    run_video_mode(session, input_name, output_names, image_list, args)

if __name__ == "__main__":
    main()
