import torch
import torch.nn as nn
from functools import partial
from typing import Dict, Optional

from ._registry import register_model, generate_default_cfgs

__all__ = ['lane_net_18']


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 288, 640),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bilinear',
        **kwargs
    }


default_cfgs = {
    'lane_net_18': _cfg(),
}


class ParsingNetHead(nn.Module):
    """Multi-head output for lane detection parsing network."""
    
    def __init__(
        self,
        in_dim: int,
        num_grid_row: int,
        num_cls_row: int,
        num_lane_on_row: int,
        num_grid_col: int,
        num_cls_col: int,
        num_lane_on_col: int,
        mlp_mid_dim: int = 1024,
        fc_norm: bool = False,
    ):
        super().__init__()
        
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_lane_on_row = num_lane_on_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_col = num_lane_on_col
        
        # Calculate output dimensions for each head
        self.dim1 = num_grid_row * num_cls_row * num_lane_on_row  # loc_row
        self.dim2 = num_grid_col * num_cls_col * num_lane_on_col  # loc_col
        self.dim3 = 2 * num_cls_row * num_lane_on_row            # exist_row
        # self.dim4 = 2 * num_cls_col * num_lane_on_col            # exist_col
        self.dim5 = 4 * 8                                          # lane_label
        
        # Shared feature processing
        norm_layer = nn.LayerNorm if fc_norm else nn.Identity
        self.shared_fc = nn.Sequential(
            norm_layer(in_dim) if fc_norm else norm_layer(),
            nn.Linear(in_dim, mlp_mid_dim),
            nn.ReLU(inplace=True),
        )
        
        # Task-specific heads
        self.loc_row_head = nn.Linear(mlp_mid_dim, self.dim1)
        self.loc_col_head = nn.Linear(mlp_mid_dim, self.dim2)
        self.exist_row_head = nn.Linear(mlp_mid_dim, self.dim3)
        # self.exist_col_head = nn.Linear(mlp_mid_dim, self.dim4)
        self.lane_label_head = nn.Linear(mlp_mid_dim, self.dim5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, export_onnx: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features of shape (B, C)
        
        Returns:
            Dictionary containing:
                - loc_row: (B, num_grid_row, num_cls_row, num_lane_on_row)
                - loc_col: (B, num_grid_col, num_cls_col, num_lane_on_col)
                - exist_row: (B, 2, num_cls_row, num_lane_on_row)
                - exist_col: (B, 2, num_cls_col, num_lane_on_col)
                - lane_label: (B, 4, 8)
        """
        feat = self.shared_fc(x)
        
        loc_row = self.loc_row_head(feat)
        loc_row = loc_row.view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row)
        
        loc_col = self.loc_col_head(feat)
        loc_col = loc_col.view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col)
        
        exist_row = self.exist_row_head(feat)
        exist_row = exist_row.view(-1, 2, self.num_cls_row, self.num_lane_on_row)
        
        # exist_col = self.exist_col_head(feat)
        # exist_col = exist_col.view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        
        lane_label = self.lane_label_head(feat)
        lane_label = lane_label.view(-1, 4, 8)

        if export_onnx:
            return {
                'loc_row': loc_row,
                'exist_row': exist_row,
                'lane_label': lane_label,
            }
        
        return {
            'loc_row': loc_row,
            'loc_col': loc_col,
            'exist_row': exist_row,
            # 'exist_col': exist_col,
            'lane_label': lane_label,
        }


class ParsingNet(nn.Module):
    """ParsingNet: Multi-task learning network for lane detection.
    
    Args:
        backbone: backbone model name (e.g., 'resnet50', 'resnet101')
        num_grid_row: number of row grids
        num_cls_row: number of row classes
        num_lane_on_row: number of lanes on row
        num_grid_col: number of column grids
        num_cls_col: number of column classes
        num_lane_on_col: number of lanes on column
        input_height: input image height
        input_width: input image width
        mlp_mid_dim: hidden dimension of MLP (default: 1024)
        fc_norm: use LayerNorm before FC layer (default: False)
        pretrained: load pretrained backbone weights
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        num_grid_row: int = 100,
        num_cls_row: int = 58,
        num_lane_on_row: int = 4,
        num_grid_col: int = 100,
        num_cls_col: int = 65,
        num_lane_on_col: int = 4,
        input_height: int = 288,
        input_width: int = 640,
        mlp_mid_dim: int = 1024,
        fc_norm: bool = False,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_lane_on_row = num_lane_on_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_col = num_lane_on_col
        self.input_height = input_height
        self.input_width = input_width
        self.export_onnx = False
        
        from timm import create_model
        # Backbone network (ResNet)
        self.backbone = create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(4,),  # Get layer3 and layer4 outputs
        )

        feature_info = self.backbone.feature_info
        backbone_chs = feature_info.channels()[-1]
        
        # Neck: 512 → 256 → 512 (bottleneck for dimension reduction)
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_chs, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, backbone_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(backbone_chs),
        )
        
        # Pool layer: 512 → 16 channels
        tmp_dim = 16
        self.pool = nn.Sequential(
            nn.Conv2d(backbone_chs, tmp_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(tmp_dim),
            nn.ReLU(inplace=True),
        )

        training = self.pool.training

        self.pool.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_height, input_width)
            feat = self.backbone(dummy)[-1]
            feat = self.pool(feat)

        self.pool.train(training)

        self.input_dim = feat.flatten(1).shape[1]
        
        # Parsing head with multi-task outputs
        self.parsing_head = ParsingNetHead(
            in_dim=self.input_dim,
            num_grid_row=num_grid_row,
            num_cls_row=num_cls_row,
            num_lane_on_row=num_lane_on_row,
            num_grid_col=num_grid_col,
            num_cls_col=num_cls_col,
            num_lane_on_col=num_lane_on_col,
            mlp_mid_dim=mlp_mid_dim,
            fc_norm=fc_norm,
        )
        
        # Initialize neck weights
        self._init_neck_weights()
    
    def _init_neck_weights(self):
        """Initialize neck weights using Kaiming initialization."""
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.pool.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        
        Returns:
            Dictionary with lane detection predictions:
                - loc_row: row coordinate predictions
                - loc_col: column coordinate predictions
                - exist_row: row existence probability
                - exist_col: column existence probability
                - lane_label: lane class labels
        """
        # Backbone forward (returns list of features from specified layers)
        features = self.backbone(x)
        fea = features[-1]  # Use the last feature map (layer4 output, stride 32)
        
        # Apply neck (dimension reduction bottleneck)
        fea = fea + self.neck(fea)
        
        # Pool to reduce channels
        fea = self.pool(fea)
        
        # Flatten for FC layers
        fea = fea.flatten(1)
        
        # Parsing head outputs multi-task predictions
        pred_dict = self.parsing_head(fea, export_onnx=self.export_onnx)

        if self.export_onnx:
            return (
                pred_dict['loc_row'],
                # pred_dict['loc_col'],
                pred_dict['exist_row'],
                # pred_dict['exist_col'],
                pred_dict['lane_label']
            )
        
        return pred_dict


def _parsing_net(
    backbone: str,
    pretrained: bool = False,
    **kwargs,
) -> ParsingNet:
    """Create ParsingNet model."""
    model = ParsingNet(
        backbone=backbone,
        pretrained=pretrained,
        **kwargs,
    )
    return model


@register_model
def lane_net_18(pretrained: bool = False, **kwargs) -> ParsingNet:
    """ParsingNet with ResNet18 backbone."""
    model = _parsing_net(
        'resnet18',
        pretrained=pretrained,
        **kwargs,
    )

    model.default_cfg = default_cfgs['lane_net_18']
    return model