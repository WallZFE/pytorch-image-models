import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
import numpy as np

_logger = logging.getLogger(__name__)

class LaneLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.cls_loss = SoftmaxFocalLoss(2, ignore_lb=-1)
        self.cls_loss_weight = 1.0

        self.relation_loss = ParsingRelationLoss()
        self.relation_dis = ParsingRelationDis()
        self.relation_loss_weight = 0.0
        self.relation_dis_weight = 0.0

        self.cls_loss_col = SoftmaxFocalLoss(2, ignore_lb=-1)
        self.cls_loss_col_weight = 1.0

        self.cls_ext = nn.CrossEntropyLoss()
        self.cls_ext_col = nn.CrossEntropyLoss()
        self.cls_ext_weight = 1.0
        self.cls_ext_col_weight = 1.0

        self.mean_loss_row = MeanLoss()
        self.mean_loss_col = MeanLoss()
        self.mean_loss_row_weight = 0.05
        self.mean_loss_col_weight = 0.05

        self.lane_attr_loss = nn.BCEWithLogitsLoss()
        self.lane_attr_loss_weight = 1.0

    def forward(self, pred, target):
        cls_out_ext_label = (target['labels_row'] != -1).long()
        cls_out_col_ext_label = (target['labels_col'] != -1).long()

        total_loss = torch.zeros((), device=pred['loc_row'].device)
        loss_items = {}

        if self.cls_loss_weight != 0:
            loss_cur = self.cls_loss(pred['loc_row'], target['labels_row'])
            loss_items['cls_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_loss_weight

        if self.relation_loss_weight != 0:
            loss_cur = self.relation_loss(pred['loc_row'])
            loss_items['relation_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.relation_loss_weight

        if self.relation_dis_weight != 0:
            loss_cur = self.relation_dis(pred['loc_row'])
            loss_items['relation_dis'] = loss_cur.detach()
            total_loss += loss_cur * self.relation_dis_weight

        if self.cls_loss_col_weight != 0:
            loss_cur = self.cls_loss_col(pred['loc_col'], target['labels_col'])
            loss_items['cls_loss_col'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_loss_col_weight

        if self.cls_ext_weight != 0:
            loss_cur = self.cls_ext(pred['exist_row'], cls_out_ext_label)
            loss_items['cls_ext'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_ext_weight

        if self.cls_ext_col_weight != 0:
            loss_cur = self.cls_ext_col(pred['exist_col'], cls_out_col_ext_label)
            loss_items['cls_ext_col'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_ext_col_weight

        if self.mean_loss_row_weight != 0:
            loss_cur = self.mean_loss_row(pred['loc_row'], target['labels_row'])
            loss_items['mean_loss_row'] = loss_cur.detach()
            total_loss += loss_cur * self.mean_loss_row_weight

        if self.mean_loss_col_weight != 0:
            loss_cur = self.mean_loss_col(pred['loc_col'], target['labels_col'])
            loss_items['mean_loss_col'] = loss_cur.detach()
            total_loss += loss_cur * self.mean_loss_col_weight

        if self.lane_attr_loss_weight != 0:
            loss_cur = self.lane_attr_loss(pred['lane_label'], target['lane_label'].float())
            loss_items['lane_attr_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.lane_attr_loss_weight

        loss_items['total_loss'] = total_loss.detach()

        return total_loss, loss_items


def soft_nll(pred, target, ignore_index = -1):
    C = pred.shape[1]
    invalid_target_index = target==ignore_index

    ttarget = target.clone()
    ttarget[invalid_target_index] = C

    target_l = target - 1
    target_r = target + 1

    invalid_part_l = target_l == -1
    invalid_part_r = target_r == C

    invalid_target_l_index = torch.logical_or(invalid_target_index, invalid_part_l)
    target_l[invalid_target_l_index] = C

    invalid_target_r_index = torch.logical_or(invalid_target_index, invalid_part_r)
    target_r[invalid_target_r_index] = C

    supp_part_l = target.clone()
    supp_part_r = target.clone()
    supp_part_l[target!=0] = C
    supp_part_r[target!=C-1] = C

    target_onehot = torch.nn.functional.one_hot(ttarget, num_classes=C+1)
    target_onehot = target_onehot[...,:-1].permute(0,3,1,2)

    target_l_onehot = torch.nn.functional.one_hot(target_l, num_classes=C+1)
    target_l_onehot = target_l_onehot[...,:-1].permute(0,3,1,2)

    target_r_onehot = torch.nn.functional.one_hot(target_r, num_classes=C+1)
    target_r_onehot = target_r_onehot[...,:-1].permute(0,3,1,2)

    supp_part_l_onehot = torch.nn.functional.one_hot(supp_part_l, num_classes=C+1)
    supp_part_l_onehot = supp_part_l_onehot[...,:-1].permute(0,3,1,2)

    supp_part_r_onehot = torch.nn.functional.one_hot(supp_part_r, num_classes=C+1)
    supp_part_r_onehot = supp_part_r_onehot[...,:-1].permute(0,3,1,2)

    target_fusion = 0.9 * target_onehot + 0.05 * target_l_onehot + 0.05 * target_r_onehot + 0.05 * supp_part_l_onehot + 0.05 * supp_part_r_onehot
    # import pdb; pdb.set_trace()
    return -(target_fusion * pred).sum() / (target!=ignore_index).sum()

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, soft_loss = True, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self.soft_loss = soft_loss
        if not self.soft_loss:
            self.nll = nn.NLLLoss(ignore_index=ignore_lb)


    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        if self.soft_loss:
            loss = soft_nll(log_score, labels, ignore_index = self.ignore_lb)
        else:
            loss = self.nll(log_score, labels)

        # import pdb; pdb.set_trace()
        return loss

class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()
    def forward(self, logits):
        n, c, h, w = logits.shape
        # 沿 h 维度做差分
        diff = logits[:, :, :-1, :] - logits[:, :, 1:, :]  # [n, c, h-1, w]
        # 正确拼接（如果需要的话）或者直接计算
        loss = torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), reduction='sum'
        )
        # 除以实际的元素总数（不含batch混淆）
        return loss / diff.numel()
    
class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            diff_list1.append(pos[:, i, :] - pos[:, i+1, :])  # shape: [n, w]

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i+1])  # ❌ 默认 reduction='mean'
        loss /= len(diff_list1) - 1
        return loss

class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = (logits.softmax(1) * grid).sum(1)
        loss = self.l1(logits, label.float())[label != -1]
        return loss.mean()