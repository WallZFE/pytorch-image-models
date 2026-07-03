import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import logging
import numpy as np

_logger = logging.getLogger(__name__)

class LaneLoss(nn.Module):

    def __init__(self, testing_mode=True):
        super().__init__()

        self.cls_loss = SoftmaxFocalLoss(2, ignore_lb=-1)
        self.cls_loss_weight = 1.0

        self.cls_loss_col = SoftmaxFocalLoss(2, ignore_lb=-1)
        self.cls_loss_col_weight = 0.3


        self.relation_loss = ParsingRelationLoss()
        self.relation_loss_weight = 0.0

        self.relation_dis = ParsingRelationDis()
        self.relation_dis_weight = 0.0

        if testing_mode:
            self.cls_ext = nn.CrossEntropyLoss()
            self.cls_ext_col = nn.CrossEntropyLoss()
        else:
            self.cls_ext = FocalLabelSmoothLoss(alpha=0.75, gamma=2.0, label_smoothing=0.05)
            self.cls_ext_col = FocalLabelSmoothLoss(alpha=0.75, gamma=2.0, label_smoothing=0.05)
        self.cls_ext_weight = 1.0
        self.cls_ext_col_weight = 0.0

        self.mean_loss_row = MeanLoss()
        self.mean_loss_col = MeanLoss()
        self.mean_loss_row_weight = 1.0
        self.mean_loss_col_weight = 0.3

        if testing_mode:
            self.lane_attr_loss = nn.BCEWithLogitsLoss()
        else:
            self.lane_attr_loss = LaneAttributeLoss(num_attrs=8)
        self.lane_attr_loss_weight = 1.0

    def forward(self, pred, target):
        cls_out_ext_label = (target['labels_row'] != -1).long()
        # cls_out_col_ext_label = (target['labels_col'] != -1).long()

        total_loss = torch.zeros((), device=pred['loc_row'].device)
        loss_items = {}

        if self.cls_loss_weight != 0:
            loss_cur = self.cls_loss(pred['loc_row'], target['labels_row'])
            loss_items['cls_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_loss_weight

        if self.cls_loss_col_weight != 0:
            loss_cur = self.cls_loss_col(pred['loc_col'], target['labels_col'])
            loss_items['cls_loss_col'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_loss_col_weight


        if self.relation_loss_weight != 0:
            loss_cur = self.relation_loss(pred['loc_row'])
            loss_items['relation_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.relation_loss_weight

        if self.relation_dis_weight != 0:
            loss_cur = self.relation_dis(pred['loc_row'])
            loss_items['relation_dis'] = loss_cur.detach()
            total_loss += loss_cur * self.relation_dis_weight


        if self.cls_ext_weight != 0:
            loss_cur = self.cls_ext(pred['exist_row'], cls_out_ext_label)
            loss_items['cls_ext'] = loss_cur.detach()
            total_loss += loss_cur * self.cls_ext_weight

        # if self.cls_ext_col_weight != 0:
        #     loss_cur = self.cls_ext_col(pred['exist_col'], cls_out_col_ext_label)
        #     loss_items['cls_ext_col'] = loss_cur.detach()
        #     total_loss += loss_cur * self.cls_ext_col_weight



        if self.mean_loss_row_weight != 0:
            loss_cur = self.mean_loss_row(pred['loc_row'], target['labels_row_float'])
            loss_items['mean_loss_row'] = loss_cur.detach()
            total_loss += loss_cur * self.mean_loss_row_weight

        if self.mean_loss_col_weight != 0:
            loss_cur = self.mean_loss_col(pred['loc_col'], target['labels_col_float'])
            loss_items['mean_loss_col'] = loss_cur.detach()
            total_loss += loss_cur * self.mean_loss_col_weight



        if self.lane_attr_loss_weight != 0:
            loss_cur = self.lane_attr_loss(pred['lane_label'], target['lane_label'].float())
            loss_items['lane_attr_loss'] = loss_cur.detach()
            total_loss += loss_cur * self.lane_attr_loss_weight

        loss_items['total_loss'] = total_loss.detach()

        return total_loss, loss_items

class FocalLabelSmoothLoss(nn.Module):
    """
    结合了 Focal Loss 和 Label Smoothing 的损失函数。
    专门用于解决正负样本极度不平衡，同时防止模型对标签过度自信。
    """
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha          # 正样本(存在)的基础权重，>0.5表示更看重正样本
        self.gamma = gamma          # Focal 聚焦参数，降低简单样本的权重
        self.label_smoothing = label_smoothing # 标签平滑系数

    def forward(self, inputs, targets):
        """
        inputs: (b, 2, 42, 4) - 模型输出的 logits
        targets: (b, 42, 4)   - 真实标签 (0=不存在, 1=存在)
        """
        # 计算带 Label Smoothing 的 Cross Entropy (不取平均，保留每个位置的 loss)
        # PyTorch 原生支持多维输入的 label_smoothing
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction='none')
        
        # 计算模型对真实类别的预测概率 (p_t)
        # 注意：这里用普通的 softmax 计算概率，不受 label_smoothing 影响
        probs = torch.softmax(inputs, dim=1) # shape: (b, 2, 42, 4)
        
        # 根据 targets 提取对应类别的概率
        # targets==1 时取通道1(存在)的概率，targets==0 时取通道0(不存在)的概率
        p_t = torch.where(targets == 1, probs[:, 1, :, :], probs[:, 0, :, :])
        
        # 计算 Focal 调制因子: (1 - p_t)^gamma
        # 如果模型预测得很准(p_t接近1)，这个因子就接近0，从而降低该样本的loss权重
        focal_weight = (1.0 - p_t) ** self.gamma
        
        # 计算 Alpha 权重因子
        # targets==1(存在) 权重为 alpha，targets==0(不存在) 权重为 1-alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        
        # 组合最终 Loss
        loss = alpha_t * focal_weight * ce_loss
        
        # 返回平均 loss
        return loss.mean()  

class LaneAttributeLoss(nn.Module):
    def __init__(self, num_attrs=8, gamma=0.7, reg_weight=0.05):
        super().__init__()
        self.num_attrs = num_attrs
        self.gamma = gamma       # Focal loss 的 gamma 参数，0 表示退化为普通 BCE
        self.reg_weight = reg_weight # 正则化项的权重
        
        # 记录历史正样本比例，用于平滑动态权重（防止某个batch碰巧没有某属性导致除零）
        self.register_buffer('running_pos_ratio', torch.ones(num_attrs) * 0.1)
        self.register_buffer('update_count', torch.tensor(0, dtype=torch.long)) # 用于冷启动
        self.momentum = 0.7 # 动量系数

    def forward(self, pred_attr, target_attr):
        """
        pred_attr: (b, 4, 8) 模型输出的 logits
        target_attr: (b, 4, 8) 真实标签 (0.0 或 1.0)
        """
        b, num_lanes, num_attrs = target_attr.shape
        assert num_attrs == self.num_attrs, f"属性数量不匹配: 期望 {self.num_attrs}, 实际 {num_attrs}"

        # ==========================================
        # 生成 Existence Mask (利用全0做Mask)
        # ==========================================
        # 存在的车道线必定有3个1，求和必定为3；不存在的车道线全0，求和为0
        # exist_mask shape: (b, 4), bool 类型
        exist_mask = target_attr.sum(dim=-1) > 0 
        
        # 如果当前 batch 没有任何存在的车道线，直接返回 0 loss
        if not exist_mask.any():
            return pred_attr.sum() * 0.0 

        # 提取出“存在的车道线”的数据，shape: (N_valid, 8)
        valid_pred = pred_attr[exist_mask]   
        valid_target = target_attr[exist_mask] 
        N_valid = valid_pred.shape[0]

        # ==========================================
        # 计算动态逐类别 Pos Weight (解决极端不平衡)
        # ==========================================
        with torch.no_grad():
            # 统计当前 batch 中每个属性的正样本数量 (shape: 8)
            batch_pos_count = valid_target.sum(dim=0) 
            # 计算当前 batch 的正样本比例 (shape: 8)
            batch_pos_ratio = batch_pos_count / N_valid

            # 支持多卡分布式训练(DDP)：同步所有卡的统计量
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(batch_pos_ratio, op=dist.ReduceOp.SUM)
                batch_pos_ratio /= dist.get_world_size()
            
            # 冷启动优化：第一个 batch 直接赋值，后续使用 EMA
            if self.update_count.item() == 0:
                self.running_pos_ratio.copy_(batch_pos_ratio)
            else:
                self.running_pos_ratio.mul_(self.momentum).add_(batch_pos_ratio, alpha=1.0 - self.momentum)
            
            self.update_count += 1
                
            # 计算 pos_weight: (1 - p) / p。限制最大权重为 50，防止梯度爆炸
            # 对于前4个属性(p很大)，权重接近1；对于后4个属性(p极小)，权重会被放大
            dynamic_pos_weight = torch.clamp((1.0 - self.running_pos_ratio) / (self.running_pos_ratio + 1e-6), min=1.0, max=50.0) 

        # 基础 BCE Loss (不 reduction，保留每个样本每个属性的 loss)
        # bce_loss shape: (N_valid, 8)
        bce_loss = F.binary_cross_entropy_with_logits(valid_pred, valid_target, pos_weight=dynamic_pos_weight, reduction='none')

        # ==========================================
        # 融合 Focal Loss 思想 (关注难样本)
        # ==========================================
        if self.gamma > 0:
            with torch.no_grad():
                probs = torch.sigmoid(valid_pred).clamp(min=1e-7, max=1.0 - 1e-7)
                # p_t: 预测正确的概率 (标签为1取p，标签为0取1-p)
                p_t = probs * valid_target + (1 - probs) * (1 - valid_target)
                # focal_weight: 降低容易分类样本的权重
                # 注：标准 Focal Loss 包含 alpha 平衡因子，但此处已通过 dynamic_pos_weight 
                # 解决了正负样本不平衡，因此省略 alpha，避免权重重复叠加。
                focal_weight = (1 - p_t) ** self.gamma
            bce_loss = bce_loss * focal_weight

        # 对属性和样本求平均，得到最终的分类 Loss
        loss_cls = bce_loss.mean()

        # ==========================================
        # 先验正则化 (约束概率和趋近于 3)
        # ==========================================
        # 只对存在的车道线计算正则化
        probs = torch.sigmoid(valid_pred) # (N_valid, 8)
        sum_probs = probs.sum(dim=-1)     # (N_valid,)
        
        # 既然存在的车道线必定有 3 个 1，那么概率和应该趋近于 3.0
        # 使用 Smooth L1 或 L1 惩罚偏离 3.0 的程度
        target_sum = valid_target.sum(dim=-1) 
        loss_reg = F.mse_loss(sum_probs, target_sum)

        # ==========================================
        # 综合 Loss
        # ==========================================
        total_loss = loss_cls + self.reg_weight * loss_reg
        
        return total_loss


def soft_nll(pred, target, ignore_index = -1):

    # 把target中的-1替换为C
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