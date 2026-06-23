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
            **kwargs,
    ):
        super(LaneDataset, self).__init__()
        self.root           = root
        self.split          = split
        
        if row_anchor is None or col_anchor is None:
            _logger.error("LaneDataset anchors cannot be None!")
            exit(-1)
            
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
            self.list = f.readlines()

        with open(cache_path, 'r') as f:
            self.cached_points = json.load(f)

        # TODO 增加新的数据增强
        self.aug = A.Compose([
                A.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-6, 6),
                    translate_px={"x": (-40, 40), "y": (-30, 30)},
                    fit_output=False,
                    p=1.0,
                ),
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape

        # 读取坐标
        infos = self.cached_points[img_name]
        # points shape: (num_lanes, num_points, 2) -> 通常是 (4, N, 2)
        points = np.array(infos["points"]).astype(np.float32)
        lane_label = np.array(infos["lane_label"]).astype(np.float32)
        
        # 数据增强与归一化
        img, points = self._transform(img, points)

        # 初始化 target 字典的所有键
        target = {
            "labels_row": None,
            "labels_col": None,
            "labels_row_float": None,
            "labels_col_float": None,
            "lane_label": torch.from_numpy(lane_label),
            "img_w": img_w,
            "img_h": img_h,
        }

        # row 行
        # points_row shape: (num_lanes, H_row, 2)
        points_row = self._my_interp_cpu(points, self.interp_loc_row, direction=0)
        if self.split == 'train':
            # points_row_extend shape: (H_row, num_lanes)
            points_row_extend = self._extend(points_row[:, :, 0]).transpose(0, 1) 
        else:
            # points_row_extend shape: (H_row, num_lanes)
            points_row_extend = torch.from_numpy(points_row[:, :, 0]).transpose(0, 1) 

        labels_row = (points_row_extend / img_w * (self.num_cell_row - 1)).long()
        labels_row[(points_row_extend < 0) | (points_row_extend > img_w)] = -1
        labels_row[(labels_row < 0) | (labels_row > (self.num_cell_row - 1))] = -1


        # col 列
        points_col = self._my_interp_cpu(points, self.interp_loc_col, direction=1)
        # (H_col, num_lanes)
        points_col_y = torch.from_numpy(points_col[:, :, 1]).transpose(0, 1)

        labels_col = (points_col_y / img_h * (self.num_cell_col - 1)).long()
        labels_col[(points_col_y < 0) | (points_col_y > img_h)] = -1
        labels_col[(labels_col < 0) | (labels_col > (self.num_cell_col - 1))] = -1

        labels_row_float = points_row_extend / img_w
        labels_row_float[(labels_row_float < 0) | (labels_row_float > 1)] = -1
        
        labels_col_float = points_col_y / img_h
        labels_col_float[(labels_col_float < 0) | (labels_col_float > 1)] = -1

        target["labels_row"] = labels_row
        target["labels_col"] = labels_col
        target["labels_row_float"] = labels_row_float
        target["labels_col_float"] = labels_col_float

        if self.split != 'train':
            # 1. Row Coords (选取 lane_idx = [1, 2])
            pts_row = points[[1, 2], :, :]  # (2, num_points, 2)
            interp_row = self._my_interp_cpu(pts_row, self.interp_loc_row, direction=0) # (2, H_row, 2)
            row_coords = torch.from_numpy(interp_row).float()
            
            invalid_x = (row_coords[..., 0] < 0) | (row_coords[..., 0] > img_w)
            row_coords[..., 0] = torch.where(invalid_x, torch.full_like(row_coords[..., 0], -10000.0), row_coords[..., 0])
            
            # 2. Col Coords (选取 lane_idx = [0, 3])
            pts_col = points[[0, 3], :, :]  # (2, num_points, 2)
            interp_col = self._my_interp_cpu(pts_col, self.interp_loc_col, direction=1) # (2, H_col, 2)
            col_coords = torch.from_numpy(interp_col).float()
            
            invalid_y = (col_coords[..., 1] < 0) | (col_coords[..., 1] > img_h)
            col_coords[..., 1] = torch.where(invalid_y, torch.full_like(col_coords[..., 1], -10000.0), col_coords[..., 1])

            target["row_coords"] = row_coords
            target["col_coords"] = col_coords

        return img, target

    def __len__(self):
        return len(self.list)

    def _transform(self, img, points):
        lane_num, point_num, _ = points.shape
        if self.split == 'train':
            keypoints = points.reshape(-1, 2).tolist()
            transformed = self.aug(image=img, keypoints=keypoints)
            img = transformed["image"]
            points = np.array(transformed["keypoints"], dtype=np.float32).reshape(lane_num, point_num, 2)
        
        img = self.norm(image=img)["image"]
        return img, points

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
            if lane[-1] > 0:
                continue

            valid = lane > 0
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