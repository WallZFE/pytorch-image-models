import torch
import numpy as np
import math

def process_row_lanes(loc_row, valid_row, max_idx_row, lane_indices, row_anchor_t, img_w, num_grid_row, H_row, local_width):
    device = loc_row.device
    B = loc_row.shape[0]
    num_lanes = len(lane_indices)
    lane_idx_t = torch.tensor(lane_indices, device=device)

    v = valid_row[:, :, lane_idx_t]
    m = max_idx_row[:, :, lane_idx_t]

    if v.shape[1] >= H_row:
        v = v[:, :H_row, :]
        m = m[:, :H_row, :]
    else:
        pad_h = H_row - v.shape[1]
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_h), value=False)
        m = torch.nn.functional.pad(m, (0, 0, 0, pad_h), value=0)

    v = v.bool()
    m = m.long()

    offsets = torch.arange(-local_width, local_width + 1, device=device)
    all_ind = m.unsqueeze(-1) + offsets.view(1, 1, 1, -1)
    all_ind = all_ind.clamp(0, num_grid_row - 1)

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_idx = torch.arange(H_row, device=device).view(1, H_row, 1, 1)
    l_idx = lane_idx_t.view(1, 1, num_lanes, 1)

    b_e = b_idx.expand(B, H_row, num_lanes, all_ind.shape[-1])
    h_e = h_idx.expand(B, H_row, num_lanes, all_ind.shape[-1])
    l_e = l_idx.expand(B, H_row, num_lanes, all_ind.shape[-1])

    logits = loc_row[b_e, all_ind, h_e, l_e]
    weights = logits.softmax(dim=-1)
    refined = (weights * all_ind.float()).sum(dim=-1) + 0.5

    x = refined / (num_grid_row - 1) * img_w
    x = x.permute(0, 2, 1)
    y = row_anchor_t.view(1, 1, H_row).expand(B, num_lanes, H_row)

    coords = torch.stack([x, y], dim=-1)

    invalid = ~v.permute(0, 2, 1)
    coords[..., 1] = torch.where(invalid, y.float(), coords[..., 1])
    coords[..., 0] = torch.where(invalid, torch.full_like(coords[..., 0], -10000.0), coords[..., 0])

    return coords

def process_col_lanes(loc_col, valid_col, max_idx_col, lane_indices, col_anchor_t, img_h, num_grid_col, H_col, local_width):
    device = loc_col.device
    B = loc_col.shape[0]
    num_lanes = len(lane_indices)
    lane_idx_t = torch.tensor(lane_indices, device=device)

    v = valid_col[:, :, lane_idx_t]
    m = max_idx_col[:, :, lane_idx_t]

    if v.shape[1] >= H_col:
        v = v[:, :H_col, :]
        m = m[:, :H_col, :]
    else:
        pad_h = H_col - v.shape[1]
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_h), value=False)
        m = torch.nn.functional.pad(m, (0, 0, 0, pad_h), value=0)

    v = v.bool()
    m = m.long()

    offsets = torch.arange(-local_width, local_width + 1, device=device)
    all_ind = m.unsqueeze(-1) + offsets.view(1, 1, 1, -1)
    all_ind = all_ind.clamp(0, num_grid_col - 1)

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1)
    h_idx = torch.arange(H_col, device=device).view(1, H_col, 1, 1)
    l_idx = lane_idx_t.view(1, 1, num_lanes, 1)

    b_e = b_idx.expand(B, H_col, num_lanes, all_ind.shape[-1])
    h_e = h_idx.expand(B, H_col, num_lanes, all_ind.shape[-1])
    l_e = l_idx.expand(B, H_col, num_lanes, all_ind.shape[-1])

    logits = loc_col[b_e, all_ind, h_e, l_e]
    weights = logits.softmax(dim=-1)
    refined = (weights * all_ind.float()).sum(dim=-1) + 0.5

    y = refined / (num_grid_col - 1) * img_h
    y = y.permute(0, 2, 1)
    x = col_anchor_t.view(1, 1, H_col).expand(B, num_lanes, H_col)

    coords = torch.stack([x, y], dim=-1)

    invalid = ~v.permute(0, 2, 1)
    coords[..., 0] = torch.where(invalid, x.float(), coords[..., 0])
    coords[..., 1] = torch.where(invalid, torch.full_like(coords[..., 1], -10000.0), coords[..., 1])

    return coords

def pred2coords(pred, row_anchor, col_anchor, image_widths, image_heights, local_width=1):
    device = pred['loc_row'].device
    B = pred['loc_row'].shape[0]
    num_grid_row = pred['loc_row'].shape[1]
    num_grid_col = pred['loc_col'].shape[1]
    H_row = len(row_anchor)
    H_col = len(col_anchor)

    row_anchor_t = torch.tensor(row_anchor, dtype=torch.float32, device=device)
    col_anchor_t = torch.tensor(col_anchor, dtype=torch.float32, device=device)

    if isinstance(image_widths, torch.Tensor):
        img_w = image_widths.float().to(device).reshape(-1)
        if img_w.numel() == 1:
            img_w = img_w.expand(B)
        img_w = img_w.view(B, 1, 1)
    elif isinstance(image_widths, (int, float)):
        img_w = torch.full((B, 1, 1), float(image_widths), dtype=torch.float32, device=device)
    else:
        # list / tuple
        img_w = torch.tensor(image_widths, dtype=torch.float32, device=device).reshape(-1)
        if img_w.numel() == 1:
            img_w = img_w.expand(B)
        img_w = img_w.view(B, 1, 1)

    if isinstance(image_heights, torch.Tensor):
        img_h = image_heights.float().to(device).reshape(-1)
        if img_h.numel() == 1:
            img_h = img_h.expand(B)
        img_h = img_h.view(B, 1, 1)
    elif isinstance(image_heights, (int, float)):
        img_h = torch.full((B, 1, 1), float(image_heights), dtype=torch.float32, device=device)
    else:
        img_h = torch.tensor(image_heights, dtype=torch.float32, device=device).reshape(-1)
        if img_h.numel() == 1:
            img_h = img_h.expand(B)
        img_h = img_h.view(B, 1, 1)

    max_idx_row = pred['loc_row'].argmax(dim=1)
    max_idx_col = pred['loc_col'].argmax(dim=1)

    valid_row = pred['exist_row'].argmax(1)
    valid_col = pred['exist_col'].argmax(1)

    lane_label = None
    if 'lane_label' in pred and pred['lane_label'] is not None:
        lane_label = (pred['lane_label'].sigmoid() > 0.5)  # (B, 4, 8) bool tensor

    row_lane_idx = [1, 2]
    # 返回的 row_coords 是 GPU Tensor: (B, 2, H_row, 2)
    row_coords = process_row_lanes(pred['loc_row'], valid_row, max_idx_row, row_lane_idx, row_anchor_t, img_w, num_grid_row, H_row, local_width)

    col_lane_idx = [0, 3]
    # 返回的 col_coords 是 GPU Tensor: (B, 2, H_col, 2)
    col_coords = process_col_lanes(pred['loc_col'], valid_col, max_idx_col, col_lane_idx, col_anchor_t, img_h, num_grid_col, H_col, local_width)

    return row_coords, col_coords, lane_label

def lane_test(pred, gt, row_anchor, col_anchor, train_width, train_height):
    """
    全 GPU Tensor 向量化评估。无 for 循环，速度极快。
    """
    # 1. 获取预测结果 (全部为 GPU Tensor)
    row_coords, col_coords, lane_label = pred2coords(pred, row_anchor, col_anchor, train_width, train_height)
    
    # 2. 获取 GT (确保是 Tensor 且在同一个 device 上)
    device = row_coords.device
    gt_row_coords = gt["row_coords"] if isinstance(gt["row_coords"], torch.Tensor) else torch.tensor(gt["row_coords"], device=device)
    gt_col_coords = gt["col_coords"] if isinstance(gt["col_coords"], torch.Tensor) else torch.tensor(gt["col_coords"], device=device)
    gt_lane_label = gt["lane_label"] if isinstance(gt["lane_label"], torch.Tensor) else torch.tensor(gt["lane_label"], device=device)

    B = row_coords.shape[0]

    # ================= 1. Lane Label 评估 =================
    p_lab = lane_label.bool()      # (B, 4, 8)
    g_lab = gt_lane_label.bool()   # (B, 4, 8)

    # 指标 A: 车道线级准确率 (8个属性全对才算1)
    lane_match = torch.all(p_lab == g_lab, dim=-1)
    ll_lane_correct = lane_match.sum().float()
    ll_lane_total = float(B * 4)

    # 指标 B: 属性级准确率 (多标签二分类)
    ll_attr_tp = (p_lab & g_lab).sum().float()
    ll_attr_fp = (p_lab & ~g_lab).sum().float()
    ll_attr_fn = (~p_lab & g_lab).sum().float()
    ll_attr_tn = (~p_lab & ~g_lab).sum().float()

    # ================= 2. Row Coords 评估 (关注 x 坐标, 索引 0) =================
    p_row_x = row_coords[..., 0]      # (B, 2, H_row)
    g_row_x = gt_row_coords[..., 0]   # (B, 2, H_row)
    
    p_inv_r = (p_row_x <= -2.0)
    g_inv_r = (g_row_x <= -2.0)
    
    both_valid_r = (~p_inv_r) & (~g_inv_r)
    diff_ok_r = torch.abs(p_row_x - g_row_x) <= math.ceil(train_width * 0.01)
    
    row_tp = (both_valid_r & diff_ok_r).sum().float()
    row_fp = ((~p_inv_r) & g_inv_r).sum().float()
    row_fn = (p_inv_r & (~g_inv_r)).sum().float()
    
    both_valid_wrong_r = both_valid_r & (~diff_ok_r)
    row_fp += both_valid_wrong_r.sum().float()
    row_fn += both_valid_wrong_r.sum().float()

    # ================= 3. Col Coords 评估 (关注 y 坐标, 索引 1) =================
    p_col_y = col_coords[..., 1]      # (B, 2, H_col)
    g_col_y = gt_col_coords[..., 1]   # (B, 2, H_col)
    
    p_inv_c = (p_col_y <= -2.0)
    g_inv_c = (g_col_y <= -2.0)
    
    both_valid_c = (~p_inv_c) & (~g_inv_c)
    diff_ok_c = torch.abs(p_col_y - g_col_y) <= math.ceil(train_height * 0.01)
    
    col_tp = (both_valid_c & diff_ok_c).sum().float()
    col_fp = ((~p_inv_c) & g_inv_c).sum().float()
    col_fn = (p_inv_c & (~g_inv_c)).sum().float()
    
    both_valid_wrong_c = both_valid_c & (~diff_ok_c)
    col_fp += both_valid_wrong_c.sum().float()
    col_fn += both_valid_wrong_c.sum().float()

    # ================= 只返回原始计数 =================
    results = {
        'll_lane_correct': ll_lane_correct.item(),
        'll_lane_total':   ll_lane_total,

        'll_attr_tp': ll_attr_tp.item(),
        'll_attr_fp': ll_attr_fp.item(),
        'll_attr_fn': ll_attr_fn.item(),
        'll_attr_tn': ll_attr_tn.item(),

        'row_tp': row_tp.item(),
        'row_fp': row_fp.item(),
        'row_fn': row_fn.item(),

        'col_tp': col_tp.item(),
        'col_fp': col_fp.item(),
        'col_fn': col_fn.item(),
    }
    return results

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

    # Col
    col_pr, col_re, col_f1 = _calc_f1(c['col_tp'], c['col_fp'], c['col_fn'])

    # Total
    lane_total_f1 = 0.2 * ll_attr_f1 + 0.4 * row_f1 + 0.4 * col_f1

    return {
        'll_lane_acc':       ll_lane_acc,
        'll_attr_acc':       ll_attr_acc,
        'll_attr_precision': ll_attr_pr,
        'll_attr_recall':    ll_attr_re,
        'll_attr_f1':        ll_attr_f1,
        'row_precision':     row_pr,
        'row_recall':        row_re,
        'row_f1':            row_f1,
        'col_precision':     col_pr,
        'col_recall':        col_re,
        'col_f1':            col_f1,
        'lane_total_f1':     lane_total_f1,
    }