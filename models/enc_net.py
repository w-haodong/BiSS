import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# LayerNorm2d 保持不变，是通用模块
class LayerNorm2d(nn.LayerNorm):
    """对 [B, C, H, W] 的二维特征图按通道做归一化"""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1) # 转换为 [B, H, W, C]
        x = super().forward(x)
        return x.permute(0, 3, 1, 2) # 恢复为 [B, C, H, W]

# SS2DBlock 保持不变，与论文  命名一致
class SS2DBlock(nn.Module):
    def __init__(self, dim, d_state=16, ssm_ratio=2.0, channel_first=True):
        super().__init__()
        # 动态导入SS2D以避免依赖问题
        try:
            from models.mamba.vmamba import SS2D
            self.ss2d = SS2D(d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, channel_first=channel_first)
        except ImportError:
            print("警告: 未找到 models.mamba.vmamba.SS2D。将使用占位符 nn.Identity()")
            self.ss2d = nn.Identity()
            
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        # 简化结构以匹配原始代码
        return self.ss2d(x)

#==============================================================================
# 更新后的命名与 BiSS-Net 论文对齐
#==============================================================================

class LocalInteractionEnhancer(nn.Module):
    """
    局部交互增强器 (Local Interaction Enhancer - LIE)
    
    与论文 2.1.2 节中的 LIE 对应 。
    使用倒置瓶颈结构 (Inverted Bottleneck) 增强每个 patch 内部的细粒度特征。
    """
    def __init__(self,
                 dim: int,
                 kernel_size: int = 3,
                 expansion_ratio: float = 4.0,
                 norm_layer=LayerNorm2d,
                 act_layer=nn.ReLU,
                 drop_prob: float = 0.0,
                 use_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.expansion_ratio = expansion_ratio
        hidden_dim = int(dim * expansion_ratio)

        # 1. 深度卷积 (DWConv)
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=use_bias)
        self.norm = norm_layer(dim)
        
        # 2. 逐点卷积 - 扩展 (PWConv1)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=use_bias)
        self.act = act_layer()
        
        # 3. 逐点卷积 - 压缩 (PWConv2)
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=use_bias)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input_tensor + self.drop_path(x)
        return x

class RegionalInteractionEnhancer(nn.Module):
    """
    区域交互增强器 (Regional Interaction Enhancer - RIE)
    
    与论文 2.1.2 节中的 RIE 对应 。
    使用带步长的深度可分离卷积进行下采样，并建模区域依赖关系。
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 norm_layer=LayerNorm2d,
                 act_layer=nn.ReLU,
                 use_bias: bool = False):
        super().__init__()
        assert stride > 1, "RegionalInteractionEnhancer 用于下采样 (stride > 1)"
        padding = kernel_size // 2

        # 1. 深度卷积，步长为2，用于下采样和空间交互 (DWConv_s=2)
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, groups=in_channels, bias=use_bias)
        self.norm_dw = norm_layer(in_channels)
        
        # 2. 逐点卷积，用于通道投影 (PWConv)
        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)
        self.norm_pw = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x)
        x = self.norm_dw(x)
        x = self.pwconv(x)
        x = self.norm_pw(x)
        x = self.act(x)
        return x

class BiSS_Block(nn.Module):
    """
    Bi-Scope State-Space (BiSS) Block
    
    与论文 2.1.1 节中的 BiSS 块对应 。
    这是编码器每个阶段的核心构建模块，它结合了 Bi-Scope Contextual Module (BCM) 
    和 SS2D 块，以同时捕捉局部和全局依赖。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 d_state,
                 ssm_ratio,
                 ssm_depth=2,
                 intra_kernel_size=3,
                 intra_expansion_ratio=4.0,
                 intra_drop_prob=0.0,
                 inter_kernel_size=3,
                 act_layer=nn.SiLU,
                 channel_first=False):
        super().__init__()

        # 1. Bi-Scope Contextual Module (BCM) 的第一部分：区域交互
        # 对应论文中的 Regional Interaction Enhancer (RIE)
        self.rie = RegionalInteractionEnhancer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=inter_kernel_size,
            stride=2, # 下采样
            norm_layer=LayerNorm2d,
            act_layer=act_layer
        )

        # 2. Bi-Scope Contextual Module (BCM) 的第二部分：局部交互
        # 对应论文中的 Local Interaction Enhancer (LIE)
        self.linear_in = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False) # 论文中的Linear层
        self.lie = LocalInteractionEnhancer(
            dim=out_channels,
            kernel_size=intra_kernel_size,
            expansion_ratio=intra_expansion_ratio,
            norm_layer=LayerNorm2d,
            act_layer=act_layer,
            drop_prob=intra_drop_prob,
            use_bias=False
        )

        # 3. SS2D 堆叠块，用于捕捉长距离依赖
        self.ss2d_stack = nn.Sequential(*[
            SS2DBlock(
                dim=out_channels,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                channel_first=channel_first
            ) for _ in range(ssm_depth)
        ])
        
        # 4. 最终的归一化层
        self.final_norm = LayerNorm2d(out_channels)

    def forward(self, x):
        # 步骤 1: RIE (下采样和区域交互) -> X_RIE
        x_rie = self.rie(x)

        # 步骤 2: LIE (局部特征增强)
        x_lin = self.linear_in(x_rie) # 对应论文中的 X_lin = Linear(X_RIE)
        x_lie = self.lie(x_lin)      # 对应论文中的 X_LIE

        # 步骤 3: SS2D 栈 (长距离依赖建模) -> X_SS2D
        x_ss2d = self.ss2d_stack(x_lie)
        
        # 步骤 4: 残差连接与最终归一化
        # 对应论文公式 (5): X_BiSS = Norm(X_RIE + X_SS2D)
        # 注意：您的原始代码中残差是 x_lin + x_ss2d，这与论文的 x_rie + x_ss2d 有细微差别。
        # 这里我遵循论文的结构。如果需要完全匹配您的原始逻辑，应改为 x = x_lin + x_ss2d
        x = x_rie + x_ss2d
        
        x = self.final_norm(x)
        return x

class Stem(nn.Module):
    """
    Stem 模块
    与论文 2.1.1 节中的 Stem 模块对应 。
    作为初始的轻量级特征提取器。
    """
    def __init__(self, in_channels=3, out_channels=64, act_layer=nn.ReLU):
        super().__init__()
        # 对应论文中描述的 "三个标准 3x3 卷积，其中第一个步长为2" [cite: 90, 91]
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_layer()
        
        # 论文中第三个标准卷积后紧跟两个深度卷积 [cite: 90]
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = act_layer()

        # 对应论文中描述的 "两个深度 3x3 卷积" 和 "残差连接" [cite: 90, 92]
        self.dwconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn_dw1 = nn.BatchNorm2d(out_channels)
        self.act_dw1 = act_layer()

        self.dwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn_dw2 = nn.BatchNorm2d(out_channels)
        self.act_dw2 = act_layer()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        # 带有残差连接的深度卷积部分
        identity = x
        x = self.dwconv1(x)
        x = self.bn_dw1(x)
        x = self.act_dw1(x)

        x = self.dwconv2(x)
        x = self.bn_dw2(x)
        x = self.act_dw2(x)
        x = x + identity
        return x

class BiSS_Net_Encoder(nn.Module):
    """
    BiSS-Net 的编码器部分。
    由一个 Stem 模块和四个连续的 BiSS 阶段 (Stage) 组成 [cite: 85, 201]。
    """
    def __init__(self, cfg_list, in_channels=3, channel_first=True, device="cuda:0", channels=[3, 64, 64, 128, 256, 512], act_layer=nn.ReLU):
        super().__init__()
        self.device = torch.device(device)
        self.features = nn.ModuleList()
        self.feature_channels = [channels[1]]

        self.stem = Stem(in_channels, out_channels=self.feature_channels[-1], act_layer=act_layer)

        for cfg in cfg_list:
            stage = BiSS_Block(
                in_channels=self.feature_channels[-1],
                out_channels=cfg.out_ch,
                ssm_depth=cfg.ssm_depth,
                d_state=cfg.d_state,
                ssm_ratio=cfg.ssm_ratio,
                channel_first=channel_first,
                intra_kernel_size=3,
                intra_expansion_ratio=4.0,
                intra_drop_prob=0.0,
                inter_kernel_size=3,
                act_layer=act_layer
            )
            self.features.append(stage)
            self.feature_channels.append(cfg.out_ch)

        self._init_weights()
        self.to(self.device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.to(self.device)
        x = torch.nan_to_num(x, nan=0.0)
        x = self.stem(x)
        feats = []
        for layer in self.features:
            x = layer(x)
            feats.append(x)
        return feats

class BiSS_Block_Config:
    """配置 BiSS_Block (阶段) 的参数。"""
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 ssm_depth: int,
                 d_state=32,
                 ssm_ratio=2.0):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ssm_depth = ssm_depth
        self.d_state = d_state
        self.ssm_ratio = ssm_ratio

def biss_encoder(device="cuda:0", channels=[3, 64, 64, 128, 256, 512]):
    """工厂函数，用于创建和配置 BiSS_Net_Encoder。"""
    assert len(channels) >= 6, "通道列表需要至少6个元素以匹配配置。"
    
    # BiSS 编码器的配置，每个阶段对应一个 BiSS_Block
    encoder_cfg = [
        BiSS_Block_Config(in_ch=channels[1], out_ch=channels[2], ssm_depth=1, d_state=channels[2] // 4),
        BiSS_Block_Config(in_ch=channels[2], out_ch=channels[3], ssm_depth=1, d_state=channels[3] // 4),
        BiSS_Block_Config(in_ch=channels[3], out_ch=channels[4], ssm_depth=1, d_state=channels[4] // 4),
        BiSS_Block_Config(in_ch=channels[4], out_ch=channels[5], ssm_depth=1, d_state=channels[5] // 4),
    ]
    model = BiSS_Net_Encoder(encoder_cfg, in_channels=channels[0], device=device, channels=channels, act_layer=nn.SiLU)
    return model

# ====================== 验证测试 ======================
if __name__ == '__main__':
    # 使用新的工厂函数创建模型
    model = biss_encoder(channels=[3, 64, 128, 256, 512, 1024])
    x = torch.randn(2, 3, 1024, 512)
    features = model(x)
    print("特征金字塔输出:")
    for i, feat in enumerate(features):
        print(f"阶段 {i+1}: {feat.shape}")