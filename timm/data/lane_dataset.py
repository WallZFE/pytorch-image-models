import logging
import json
import os
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT

import torch
import torch.utils.data as data

_logger = logging.getLogger(__name__)

class LaneDataset(data.Dataset):
    def __init__(
            self,
            root,
            row_anchor,
            col_anchor,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            train_height=360,
            train_width=288,
            top_crop=0.8,
            split='train',
            num_cell_row=100,
            num_cell_col=100,
            debug_ratio=0.01,
            debug_max=10,
            debug_dir="output/debug_vis",
            **kwargs,
    ):
        super(LaneDataset, self).__init__()
        self.root   = root
        self.split  = split
        self.mean   = mean
        self.std    = std
        self.train_height   = train_height
        self.train_width    = train_width
        self.top_crop       = top_crop

        if not 0.0 <= debug_ratio <= 1.0:
            raise ValueError(f"debug_ratio must be in [0, 1], got {debug_ratio}")
        if debug_max < 0:
            raise ValueError(f"debug_max must be non-negative, got {debug_max}")

        self.debug_ratio    = debug_ratio
        self.debug_max      = debug_max
        self.debug_dir      = os.fspath(debug_dir)
        self.debug_count    = 0
        
        if row_anchor is None or col_anchor is None:
            raise ValueError("LaneDataset row_anchor and col_anchor cannot be None")
            
        self.interp_loc_row = np.array(row_anchor, dtype=np.float32)
        self.interp_loc_col = np.array(col_anchor, dtype=np.float32)
        self.num_cell_row   = num_cell_row
        self.num_cell_col   = num_cell_col

        if self.split == 'train':
            list_path = os.path.join(self.root, 'train_gt.txt')
            cache_path = os.path.join(self.root, 'tusimple_anno_cache.json')
        else:
            list_path = os.path.join(self.root, 'test.txt')
            cache_path = os.path.join(self.root, 'tusimple_anno_cache_test.json')

        with open(list_path, 'r') as f:
            self.list = [line for line in f if line.strip()]

        with open(cache_path, 'r') as f:
            self.cached_points = json.load(f)

        self.aug = A.Compose([
                A.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    rotate=(-7, 7),
                    translate_px={"x": (-10, 10), "y": (-25, 25)},
                    fit_output=False,
                    p=0.4,
                ),
                # A.Perspective(
                #     scale=(0.02, 0.05),
                #     p=0.25,
                # ),
                # A.OpticalDistortion(
                #     distort_limit=0.03,
                #     p=0.15,
                # ),

                # 光照增强
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.CLAHE(),
                ], p=0.15),

                # 颜色增强
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=10,
                    val_shift_limit=5,
                    p=0.1,
                ),

                # 模糊/噪声
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.GaussNoise(noise_scale_factor=0.1),
                ], p=0.15),

                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(10, 40),
                    hole_width_range=(30, 90),
                    fill=114,
                    p=0.05,
                )
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        # 归一化与 Resize (训练和测试都使用)
        self.norm = A.Compose([
                A.Resize(height=int(train_height / top_crop), width=train_width, interpolation=cv2.INTER_LINEAR),
                A.Crop(x_min=0, y_min=int(train_height / top_crop) - train_height, x_max=train_width, y_max=int(train_height / top_crop)),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
        ])

    def __getitem__(self, index):
        line = self.list[index]
        img_name = line.split()[0].strip() if self.split == 'train' else line.strip()
        
        # 读取图片
        img_path = os.path.join(self.root, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError( f"Failed to read lane image at index {index}: {img_path}" )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # 读取坐标
        try:
            infos = self.cached_points[img_name]
        except KeyError as exc:
            raise KeyError( f"Annotation cache does not contain image {img_name!r} at index {index}" ) from exc
        # points shape: (num_lanes, num_points, 2) -> 通常是 (4, N, 2) 并且y从小到大排序
        points = np.asarray(infos["points"], dtype=np.float32)
        lane_label = np.asarray(infos["lane_label"], dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError( f"Invalid points shape for {img_name!r}: expected [num_lanes, num_points, 2], " f"got {points.shape}")
        if lane_label.ndim != 2 or lane_label.shape[0] != points.shape[0]:
            raise ValueError( f"Invalid lane_label shape for {img_name!r}: points has {points.shape[0]} lanes, " f"lane_label has shape {lane_label.shape}")
        if points.shape[0] != 4:
            raise ValueError( f"LaneDataset expects exactly 4 lanes, got {points.shape[0]} for {img_name!r}" )

        sort_indices = np.argsort(points[:, :, 1], axis=1)
        points = np.take_along_axis(points, sort_indices[:, :, np.newaxis], axis=1)

        # _transform never mutates these input arrays in-place, so debug references do not
        # require an unconditional full image/annotation copy for every training sample.
        img_raw = img
        points_raw = points
        lane_label_raw = lane_label
        
        # 数据增强与归一化
        img, points, lane_label, img_aug = self._transform(img, points, lane_label)

        if (self.split == 'train' and self.debug_count < self.debug_max and np.random.rand() < self.debug_ratio):
            self.debug_count += 1
            os.makedirs(self.debug_dir, exist_ok=True)
            save_path_raw = os.path.join(self.debug_dir, f"{os.path.splitext(img_name)[0].replace('/', '-')}_{index}.jpg")
            save_path_aug = os.path.join(self.debug_dir, f"aug_{index}.jpg")
            self.draw_lanes(cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR), points_raw.copy(), lane_label_raw.copy(), save_path_raw)
            self.draw_lanes(cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR), points.copy(), lane_label.copy(), save_path_aug)

        # 初始化 target 字典的所有键
        target = {
            "labels_row": None,
            "labels_col": None,
            "labels_row_float": None,
            "labels_col_float": None,
            "lane_label": None,
            "img_w": img_w,
            "img_h": img_h,
        }

        # row 行
        # points_row shape: (num_lanes, H_row, 2)
        points_row = self._my_interp_cpu(points, self.interp_loc_row, direction=0)
        if self.split == 'train':
            # points_row_extend shape: (H_row, num_lanes)
            # points_row_extend = self._extend(points_row[:, :, 0]).transpose(0, 1)
            points_row_extend = torch.from_numpy(points_row[:, :, 0]).transpose(0, 1)
        else:
            # Row Coords (num_lanes, H_row, 2)
            row_coords = torch.from_numpy(points_row).float()
            invalid_x = (row_coords[..., 0] < 0) | (row_coords[..., 0] >= img_w)
            row_coords[..., 0] = torch.where(invalid_x, torch.full_like(row_coords[..., 0], -10000.0), row_coords[..., 0])
            target["row_coords"] = row_coords

            # points_row_extend shape: (H_row, num_lanes)
            points_row_extend = torch.from_numpy(points_row[:, :, 0]).transpose(0, 1) 

        labels_row = (points_row_extend / img_w * (self.num_cell_row - 1)).round().long()
        labels_row[(points_row_extend < 0) | (points_row_extend >= img_w)] = -1
        labels_row[(labels_row < 0) | (labels_row > (self.num_cell_row - 1))] = -1

        # labels_row 对应车道线 0, 1, 2, 3
        # labels_row shape: (H_row, num_lanes) → 取第 0, 1, 2, 3 列
        for lane_idx in [0, 1, 2, 3]:
            if torch.all(labels_row[:, lane_idx] < 0):
                lane_label[lane_idx] = 0.0   # 整个 (X, 8) 子维度全部置零


        # col 列
        # points_col shape: (num_lanes, H_col, 2)
        points_col = self._my_interp_cpu(points, self.interp_loc_col, direction=1)
        if self.split != 'train':
            # Col Coords (num_lanes, H_col, 2)
            col_coords = torch.from_numpy(points_col).float()
            invalid_y = (col_coords[..., 1] < 0) | (col_coords[..., 1] >= img_h)
            col_coords[..., 1] = torch.where(invalid_y, torch.full_like(col_coords[..., 1], -10000.0), col_coords[..., 1])
            target["col_coords"] = col_coords

        # (H_col, num_lanes)
        points_col_y = torch.from_numpy(points_col[:, :, 1]).transpose(0, 1)

        labels_col = (points_col_y / img_h * (self.num_cell_col - 1)).round().long()
        labels_col[(points_col_y < 0) | (points_col_y >= img_h)] = -1
        labels_col[(labels_col < 0) | (labels_col > (self.num_cell_col - 1))] = -1

        # 训练阶段保留 col 标签；lane_label 有效性仍按 row 结果判断。
        # ONNX 推理阶段不使用 col。
        # # labels_col 对应车道线 0, 3
        # # labels_col shape: (H_col, num_lanes) → 取第 0, 3 列
        # for lane_idx in [0, 3]:
        #    if torch.all(labels_col[:, lane_idx] < 0):
        #         lane_label[lane_idx] = 0.0

        labels_row_float = points_row_extend / img_w * (self.num_cell_row - 1)
        labels_row_float[(labels_row_float < 0) | (labels_row_float > (self.num_cell_row - 1))] = -1
        
        labels_col_float = points_col_y / img_h * (self.num_cell_col - 1)
        labels_col_float[(labels_col_float < 0) | (labels_col_float > (self.num_cell_col - 1))] = -1

        target["labels_row"] = labels_row
        target["labels_col"] = labels_col
        target["labels_row_float"] = labels_row_float
        target["labels_col_float"] = labels_col_float
        target["lane_label"] = torch.from_numpy(lane_label)

        return img, target

    def __len__(self):
        return len(self.list)

    def _transform(self, img, points, lane_label):
        lane_num, point_num, _ = points.shape
        points = points.copy()
        lane_label = lane_label.copy()
        valid_keypoint_mask = ((points[:, :, 0] >= 0) & (points[:, :, 0] < img.shape[1]) & (points[:, :, 1] >= 0) & (points[:, :, 1] < img.shape[0]))

        if self.split == 'train':
            do_flip = np.random.random() < 0.2
            
            if do_flip:
                img = np.ascontiguousarray(img[:, ::-1, :])
                points[:, :, 0] = img.shape[1] - 1 - points[:, :, 0]
                points = np.ascontiguousarray(points[::-1])                   # 翻转车道线顺序
                valid_keypoint_mask = np.ascontiguousarray(valid_keypoint_mask[::-1])
                lane_label = np.ascontiguousarray(lane_label[::-1])            # 翻转标签
                lane_label[:, [6, 7]] = lane_label[:, [7, 6]]                  # 实虚线 <-> 虚实线

            # Only valid annotation points participate in geometric augmentation.
            # Otherwise a missing-point placeholder such as (-1, -1) can be moved
            # into the image by an affine translation and become a fake lane point.
            keypoints = points[valid_keypoint_mask].tolist()
            transformed = self.aug(image=img, keypoints=keypoints)
            img_aug  = transformed["image"]
            transformed_points = np.asarray(transformed["keypoints"], dtype=np.float32).reshape(-1, 2)
            points = np.full((lane_num, point_num, 2), -1.0, dtype=np.float32)
            points[valid_keypoint_mask] = transformed_points
        else:
            img_aug = img.copy()
        
        img_h_aug, img_w_aug = img_aug.shape[:2]
        invalid_mask = (points[:,:,0] < 0) | (points[:,:,0] >= img_w_aug) | (points[:,:,1] < 0) | (points[:,:,1] >= img_h_aug)
        points[invalid_mask] = -1.0
        for l in range(lane_num):
            if np.all(points[l, :, 0] < 0):  
                lane_label[l] = 0.0

        img = self.norm(image=img_aug)["image"]
        return img, points, lane_label, img_aug

    def _my_interp_cpu(self, points, interp_loc, direction=0):
        """
        三维张量插值 (去除了 batch 维度)
        Args:
            points: [lane_num, point_num, 2]
            interp_loc: [new_point_num]
            direction: 0 -> interpolate x by y, 1 -> interpolate y by x
        Returns:
            [lane_num, new_point_num, 2]
        """
        if direction not in (0, 1):
            raise ValueError(f"direction must be 0 or 1, got {direction}")

        lane_num, point_num, _ = points.shape
        interp_loc = np.asarray(interp_loc, dtype=np.float32)
        new_point_num = len(interp_loc)
        output = np.full((lane_num, new_point_num, 2), -1.0, dtype=np.float32)

        inv_dir = 1 - direction
        output[:, :, inv_dir] = interp_loc[None, :]

        if point_num < 2 or new_point_num == 0:
            return output

        for l in range(lane_num):
            lane = points[l]
            lane_dir = lane[:, direction]
            lane_inv = lane[:, inv_dir]

            p1_dir = lane_dir[1:]
            p0_dir = lane_dir[:-1]
            p1_inv = lane_inv[1:]
            p0_inv = lane_inv[:-1]
            valid_segments = (
                (p1_dir >= 0)
                & (p0_dir >= 0)
                & (p1_inv >= 0)
                & (p0_inv >= 0)
            )
            if not np.any(valid_segments):
                continue

            # Match the original reverse scan: if more than one segment crosses an
            # anchor, select the segment with the greatest point index.
            crossings = valid_segments[None, :] & (
                (p1_inv[None, :] - interp_loc[:, None])
                * (p0_inv[None, :] - interp_loc[:, None])
                <= 0
            )
            matched_anchor = np.any(crossings, axis=1)
            if not np.any(matched_anchor):
                continue

            matched_indices = np.flatnonzero(matched_anchor)
            reversed_segment = np.argmax(crossings[matched_anchor, ::-1], axis=1)
            segment_indices = crossings.shape[1] - 1 - reversed_segment

            selected_p1_inv = p1_inv[segment_indices]
            selected_p0_inv = p0_inv[segment_indices]
            lengths = np.abs(selected_p1_inv - selected_p0_inv)
            non_degenerate = lengths >= 1e-6
            if not np.any(non_degenerate):
                continue

            matched_indices = matched_indices[non_degenerate]
            segment_indices = segment_indices[non_degenerate]
            lengths = lengths[non_degenerate]
            current_locs = interp_loc[matched_indices]

            factor1 = 1.0 - np.abs(p1_inv[segment_indices] - current_locs) / lengths
            factor2 = 1.0 - factor1
            output[l, matched_indices, direction] = (
                p1_dir[segment_indices] * factor1
                + p0_dir[segment_indices] * factor2
            )

        return output

    def _extend(self, coords):
        """
        三维张量延伸 (去除了 batch 维度)
        Args:
            coords: [num_lanes, num_cls]
        Returns:
            [num_lanes, num_cls] (torch.Tensor)
        """
        num_lanes, num_cls = coords.shape
        coords_axis = np.arange(num_cls)
        fitted_coords = coords.copy()
        
        for j in range(num_lanes):
            lane = coords[j]
            if lane[-1] >= 0:
                continue

            valid = lane >= 0
            num_valid_pts = np.sum(valid)
            if num_valid_pts < 6:
                continue

            valid_axis = coords_axis[valid]
            valid_lane = lane[valid]
            half = num_valid_pts // 2
            
            p = np.polyfit(valid_axis[half:], valid_lane[half:], deg=1)
            start_point = valid_axis[half]
            fitted_lane = np.polyval(p, np.arange(start_point, num_cls))

            fitted_coords[j, start_point:] = fitted_lane

        return torch.from_numpy(fitted_coords.astype(np.float32))
    
    def draw_lanes(self, image, points, lane_label, save_path):
        """
        绘制车道线并优化可视化效果
        
        参数:
            image: 输入图像 (BGR格式)
            points: 车道线点坐标 [num_lanes, num_points, 2]
            lane_label: 车道线标签 [num_lanes, 8] (4条车道线 x 8个类别)
            save_path: 保存路径
        """
        vis = image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        
        # 确保 lane_label 是二维数组 (4, 8)
        lane_label = np.array(lane_label)
        if lane_label.ndim == 1:
            lane_label = lane_label.reshape(4, 8)
        
        # 计算每条车道线的标签文本
        lane_texts = []
        for i in range(lane_label.shape[0]):
            # 获取值为1的索引
            active_indices = np.where(lane_label[i] == 1)[0]
            # 生成简洁的标签文本
            text = f"L{i}: " + ", ".join([f"{idx}" for idx in active_indices])
            lane_texts.append(text)
        
        # 绘制车道线
        for lane_idx, lane in enumerate(points):
            color = colors[lane_idx % len(colors)]
            valid = ((lane[:, 0] >= 0) & (lane[:, 1] >= 0))
            pts = lane[valid].astype(np.int32)
            
            if len(pts) == 0:
                continue
                
            # 绘制点和线
            for pt in pts:
                cv2.circle(vis, tuple(pt), 3, color, -1)
            cv2.polylines(vis, [pts], False, color, 2)
            
            # 为标签添加垂直偏移避免重叠
            offset_y = lane_idx * 25  # 每条车道线向下偏移25像素
            if len(pts) > 0:
                # 在起点上方显示标签
                text_pos = (pts[0][0], max(pts[0][1] - 10 - offset_y, 30))
                cv2.putText(vis, 
                            lane_texts[lane_idx],
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA)
        
        # 等比例缩放到1920x1080（保持宽高比）
        h, w = vis.shape[:2]
        target_width = 1920
        target_height = 1080
        
        # 计算缩放比例
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 等比例缩放
        vis_resized = cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建黑色背景画布
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 将缩放后的图像居中放置
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = vis_resized
        
        # 保存最终图像
        cv2.imwrite(save_path, canvas)