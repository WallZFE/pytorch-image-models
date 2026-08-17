import torch
import torch.nn as nn
from typing import Dict, Optional

from ._registry import register_model, generate_default_cfgs

__all__ = [
    'lane_net_18',
    'lane_net_regnetx_004',
    'lane_net_resnet18',
    'lane_net_mobilenetv3_large',
]


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
    'lane_net_regnetx_004': _cfg(),
    'lane_net_resnet18': _cfg(),
    'lane_net_mobilenetv3_large': _cfg(),
}


class ParsingNetHead(nn.Module):
    """Fully convolutional multi-head output for lane detection parsing."""
    
    def __init__(
        self,
        in_dim: int,
        num_grid_row: int,
        num_cls_row: int,
        num_lane_on_row: int,
        num_grid_col: int,
        num_cls_col: int,
        num_lane_on_col: int,
        feature_height: int,
        feature_width: int,
        head_dim: int = 64,
    ):
        super().__init__()
        
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_lane_on_row = num_lane_on_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_col = num_lane_on_col
        self.num_lane_attrs = 8
        self.feature_height = feature_height
        self.feature_width = feature_width
        
        # The FPN variants operate on the stride-16 18x40 feature map. The
        # non-FPN model remains compatible with its stride-32 9x20 feature.
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
        )

        # Axis projection is a regular 1x1 Conv after moving the feature-height
        # axis to channels. It learns H -> row anchors using every input row.
        self.row_axis_proj = nn.Conv2d(feature_height, num_cls_row, kernel_size=1)
        self.row_axis_norm = nn.BatchNorm2d(num_cls_row)
        self.row_axis_act = nn.ReLU(inplace=True)

        # Both row branches consume the same row-anchor-aligned feature
        # (B, head_dim, num_cls_row, feature_width).
        self.loc_row_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, num_lane_on_row, kernel_size=1),
        )
        self.loc_grid_proj = nn.Conv2d(feature_width, num_grid_row, kernel_size=1)
        self.lane_axis_proj = nn.Conv2d(feature_width, num_lane_on_row, kernel_size=1)
        self.lane_axis_norm = nn.BatchNorm2d(num_lane_on_row)
        self.lane_axis_act = nn.ReLU(inplace=True)
        self.exist_row_head = nn.Sequential(
            nn.Conv2d(
                head_dim,
                head_dim,
                kernel_size=(3, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 2, kernel_size=1),
        )

        # Attribute prediction has its own task features and lane-slot mapping;
        # it does not reuse the existence-oriented lane_aligned feature.
        self.lane_label_stem = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
        )
        self.lane_label_lane_proj = nn.Conv2d(
            feature_width, num_lane_on_row, kernel_size=1,
        )
        self.lane_label_lane_norm = nn.BatchNorm2d(num_lane_on_row)
        self.lane_label_lane_act = nn.ReLU(inplace=True)

        # Grouped 1x1 convolutions give every lane an independent 58 -> 1 row
        # aggregation and an independent head_dim -> 8 attribute classifier.
        self.lane_label_row_proj = nn.Conv2d(
            num_lane_on_row * num_cls_row,
            num_lane_on_row,
            kernel_size=1,
            groups=num_lane_on_row,
        )
        self.lane_label_row_norm = nn.BatchNorm2d(num_lane_on_row)
        self.lane_label_row_act = nn.ReLU(inplace=True)
        self.lane_label_head = nn.Conv2d(
            num_lane_on_row * head_dim,
            num_lane_on_row * self.num_lane_attrs,
            kernel_size=1,
            groups=num_lane_on_row,
        )

        # loc_col remains an auxiliary training head, but all output positions
        # are now learned. H -> grid_col and W -> cls_col replace Bilinear Resize.
        self.loc_col_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(head_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, num_lane_on_col, kernel_size=1),
        )
        self.loc_col_grid_proj = nn.Conv2d(feature_height, num_grid_col, kernel_size=1)
        self.loc_col_cls_proj = nn.Conv2d(feature_width, num_cls_col, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, export_onnx: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features of shape (B, C, H, W)
        
        Returns:
            Dictionary containing:
                - loc_row: (B, num_grid_row, num_cls_row, num_lane_on_row)
                - loc_col: (B, num_grid_col, num_cls_col, num_lane_on_col)
                - exist_row: (B, 2, num_cls_row, num_lane_on_row)
                - lane_label: (B, 4, 8)
        """
        feat = self.shared_conv(x)

        # (B,C,H,W) -> (B,H,C,W) -> Conv(H,row_anchors) -> (B,C,row_anchors,W)
        row_aligned = feat.permute(0, 2, 1, 3).contiguous()
        row_aligned = self.row_axis_proj(row_aligned)
        row_aligned = self.row_axis_act(self.row_axis_norm(row_aligned))
        row_aligned = row_aligned.permute(0, 2, 1, 3).contiguous()

        loc_row = self.loc_row_head(row_aligned)
        loc_row = loc_row.permute(0, 3, 2, 1).contiguous()
        loc_row = self.loc_grid_proj(loc_row)

        # Shared aligned feature: W -> four learned lane slots.
        lane_aligned = row_aligned.permute(0, 3, 2, 1).contiguous()
        lane_aligned = self.lane_axis_proj(lane_aligned)
        lane_aligned = self.lane_axis_act(self.lane_axis_norm(lane_aligned))
        lane_aligned = lane_aligned.permute(0, 3, 2, 1).contiguous()

        exist_row = self.exist_row_head(lane_aligned)

        # Dedicated attribute features: (B,C,R,W) -> (B,L,R,C).
        label_feat = self.lane_label_stem(row_aligned)
        label_feat = label_feat.permute(0, 3, 2, 1).contiguous()
        label_feat = self.lane_label_lane_proj(label_feat)
        label_feat = self.lane_label_lane_act(self.lane_label_lane_norm(label_feat))

        # Each group consumes only its own lane's R anchors.
        label_feat = label_feat.reshape(
            label_feat.shape[0],
            self.num_lane_on_row * self.num_cls_row,
            1,
            label_feat.shape[-1],
        )
        lane_desc = self.lane_label_row_proj(label_feat)
        lane_desc = self.lane_label_row_act(self.lane_label_row_norm(lane_desc))
        lane_desc = lane_desc.reshape(
            lane_desc.shape[0], self.num_lane_on_row * label_feat.shape[-1], 1, 1,
        )
        lane_label = self.lane_label_head(lane_desc).reshape(
            lane_desc.shape[0], self.num_lane_on_row, self.num_lane_attrs,
        )

        if export_onnx:
            return {
                'loc_row': loc_row,
                'exist_row': exist_row,
                'lane_label': lane_label,
            }

        # Auxiliary column branch: both output axes are learned projections.
        loc_col = self.loc_col_head(feat)
        loc_col = self.loc_col_grid_proj(
            loc_col.permute(0, 2, 3, 1).contiguous()
        )
        loc_col = self.loc_col_cls_proj(
            loc_col.permute(0, 2, 1, 3).contiguous()
        ).permute(0, 2, 1, 3).contiguous()

        return {
            'loc_row': loc_row,
            'loc_col': loc_col,
            'exist_row': exist_row,
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
        head_dim: shared convolution head channels (default: 64)
        pretrained: load pretrained backbone weights
        use_fpn: fuse stride-8/stride-16/stride-32 features into a stride-16 map
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
        head_dim: int = 64,
        pretrained: bool = False,
        use_fpn: bool = False,
        **kwargs,
    ):
        super().__init__()

        if num_lane_on_row != 4 or num_lane_on_col != 4:
            raise ValueError(
                "The current dataset, attribute targets, decoder, and metrics require exactly "
                f"4 lane slots, got row={num_lane_on_row}, col={num_lane_on_col}."
            )
        
        self.backbone_name = backbone
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_lane_on_row = num_lane_on_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_col = num_lane_on_col
        self.input_height = input_height
        self.input_width = input_width
        self.export_onnx = False
        self.use_fpn = use_fpn
        
        from timm import create_model

        if backbone.startswith('resnet') and not use_fpn:
            # Backbone network (ResNet)
            self.backbone = create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(4,),  # Get layer4 output
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

            # Keep a uniform, deployment-friendly head width without a
            # separate post-neck pooling/projection layer.
            self.fuse = nn.Sequential(
                nn.Conv2d(backbone_chs, head_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(head_dim),
                nn.ReLU(inplace=True),
            )
            backbone_chs = head_dim
            feature_stride = 32

        elif backbone in ('mobilenetv3_large_100', 'regnetx_004') or (
            backbone.startswith('resnet') and use_fpn
        ):
            # Use stride-8/16/32 maps. C5 supplies semantics, C4 supplies the
            # main stride-16 map, and C3 injects lightweight spatial detail.
            self.backbone = create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                out_indices=(2, 3, 4),
            )

            feature_info = self.backbone.feature_info
            c3_chs, c4_chs, c5_chs = feature_info.channels()
            fpn_dim = 128

            # Top-down FPN: upsample stride-32 C5 by exactly 2x and add it to
            # stride-16 C4. The parsing head therefore retains the finer C4 map.
            self.lat3 = nn.Sequential(
                nn.Conv2d(c4_chs, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
            )
            self.lat4 = nn.Sequential(
                nn.Conv2d(c5_chs, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(fpn_dim, head_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(head_dim),
            )
            # Lightweight C3 detail path: stride-8 -> stride-16. It contributes
            # fine lane boundaries without running the parsing head at stride-8.
            self.detail3 = nn.Sequential(
                nn.Conv2d(c3_chs, head_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(head_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(head_dim),
            )
            self.fpn_act = nn.ReLU(inplace=True)
            self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')

            backbone_chs = head_dim
            feature_stride = 16
        else:
            raise ValueError(
                "Unsupported backbone. Expected a ResNet variant, 'regnetx_004', or "
                "'mobilenetv3_large_100', "
                f"got {backbone!r}."
            )
        
        self.feature_dim = backbone_chs
        feature_height = (input_height + feature_stride - 1) // feature_stride
        feature_width = (input_width + feature_stride - 1) // feature_stride

        # Parsing head with multi-task outputs
        self.parsing_head = ParsingNetHead(
            in_dim=self.feature_dim,
            num_grid_row=num_grid_row,
            num_cls_row=num_cls_row,
            num_lane_on_row=num_lane_on_row,
            num_grid_col=num_grid_col,
            num_cls_col=num_cls_col,
            num_lane_on_col=num_lane_on_col,
            feature_height=feature_height,
            feature_width=feature_width,
            head_dim=head_dim,
        )
        
        # Initialize neck weights
        self._init_neck_weights()
    
    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and fuse the feature map consumed by the parsing head."""
        if self.backbone_name.startswith('resnet') and not self.use_fpn:
            fea = self.backbone(x)[-1]
            return self.fuse(fea + self.neck(fea))

        c3, c4, c5 = self.backbone(x)
        p5 = self.p5_upsample(self.lat4(c5))
        semantic = self.fuse(self.lat3(c4) + p5)
        detail = self.detail3(c3)
        return self.fpn_act(semantic + detail)

    def _init_neck_weights(self):
        """Initialize neck and FPN weights using Kaiming initialization."""
        if self.backbone_name.startswith('resnet') and not self.use_fpn:
            modules = [self.neck, self.fuse]
        else:
            modules = [self.lat3, self.lat4, self.fuse, self.detail3]

        for module in modules:
            for m in module.modules():
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
        fea = self._forward_backbone(x)
        
        # Parsing head outputs multi-task predictions
        pred_dict = self.parsing_head(fea, export_onnx=self.export_onnx)

        if self.export_onnx:
            return (
                pred_dict['loc_row'],
                pred_dict['exist_row'],
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


@register_model
def lane_net_regnetx_004(pretrained: bool = False, **kwargs) -> ParsingNet:
    """ParsingNet with a quantization-friendly RegNetX-004 backbone."""
    model = _parsing_net(
        'regnetx_004',
        pretrained=pretrained,
        use_fpn=True,
        **kwargs,
    )

    model.default_cfg = default_cfgs['lane_net_regnetx_004']
    return model


@register_model
def lane_net_resnet18(pretrained: bool = False, **kwargs) -> ParsingNet:
    """ResNet18 baseline with stride-8 detail and C4/C5 FPN features."""
    model = _parsing_net(
        'resnet18',
        pretrained=pretrained,
        use_fpn=True,
        **kwargs,
    )

    model.default_cfg = default_cfgs['lane_net_resnet18']
    return model


@register_model
def lane_net_mobilenetv3_large(pretrained: bool = False, **kwargs) -> ParsingNet:
    """ParsingNet with MobileNetV3-Large backbone."""

    model = _parsing_net(
        'mobilenetv3_large_100',
        pretrained=pretrained,
        **kwargs,
    )

    model.default_cfg = default_cfgs['lane_net_mobilenetv3_large']
    return model
