import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    """
    对 [B, C, H, W] 格式的二维特征图按通道进行层归一化。
    这是一个通用的辅助模块，因此保持在外部定义。
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 将张量维度从 [B, C, H, W] 转换为 [B, H, W, C] 以匹配 nn.LayerNorm 的期望输入
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        # 将张量维度恢复为 [B, C, H, W]
        return x.permute(0, 3, 1, 2)

class DecNet(nn.Module):
    """
    解码器网络 (Decoder Network)。
    这个类现在内部集成了所有必要的子模块，以实现一个完整、独立的解码器。
    """

    class _CombinationModule(nn.Module):
        """
        内部类：用于融合来自较低层 (x_low) 和较高层 (x_up) 的特征。
        它首先对低层特征进行上采样，然后与高层特征拼接，最后通过卷积进行融合。
        """
        def __init__(self, c_low: int, c_up: int, act_layer: nn.Module, norm_layer: nn.Module):
            super().__init__()
            # 上采样并调整通道数的卷积层
            self.upsample_conv = nn.Sequential(
                nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                norm_layer(c_up),
                act_layer()
            )
            # 对拼接后的特征进行融合的卷积层
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                norm_layer(c_up),
                act_layer()
            )

        def forward(self, x_low: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
            # 使用双线性插值对低层特征进行上采样，使其与高层特征尺寸匹配
            x_low_upsampled = self.upsample_conv(
                F.interpolate(x_low, size=x_up.shape[2:], mode='bilinear', align_corners=False)
            )
            # 在通道维度上拼接高层特征和上采样后的低层特征
            x_cat = torch.cat([x_up, x_low_upsampled], dim=1)
            # 通过 1x1 卷积融合拼接后的特征
            return self.fusion_conv(x_cat)

    class _UpsampleConvBlock(nn.Module):
        """
        内部类：一个通用的上采样卷积模块。
        """
        def __init__(self, in_channels: int, out_channels: int, scale_factor: int, act_layer: nn.Module, norm_layer: nn.Module):
            super().__init__()
            self.process = nn.Sequential(
                # 1x1 卷积用于调整通道数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                act_layer(),
                # 上采样
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                # 3x3 卷积用于处理上采样后的特征
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channels),
                act_layer()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.process(x)

    class _FusionModule(nn.Module):
        """
        内部类：用于最终的多尺度特征融合。
        它接收来自不同解码阶段的特征，通过拼接和相加的方式进行融合。
        """
        def __init__(self, in_channels_per_input: int, out_channels: int, act_layer: nn.Module, norm_layer: nn.Module):
            super().__init__()
            # 最终融合模块接收4个输入拼接：c2, c3, c4, 和它们的平均值
            total_in_channels = in_channels_per_input * 4
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(total_in_channels, out_channels, kernel_size=1),
                norm_layer(out_channels),
                act_layer()
            )

        def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
            # 逐元素平均
            features_mean = (c2 + c3 + c4) / 3
            # 在通道维度上拼接所有输入特征
            features_concat = torch.cat([c2, c3, c4, features_mean], dim=1)
            return self.fusion_conv(features_concat)


    # DecNet 的主构造函数
    def __init__(self, heads: dict, channel: int, channels: list = [3, 64, 64, 128, 256, 512], act_layer: nn.Module = nn.SiLU, norm_layer: nn.Module = LayerNorm2d):
        super(DecNet, self).__init__()
        self.heads = heads
        
        # 使用内部类来构建解码器层
        # `channels` 列表: [c0, c1, c2, c3, c4, c5]
        # c2, c3, c4, c5 是骨干网络不同阶段的输出通道数
        self.dec_c4 = self._CombinationModule(channels[-1], channels[-2], act_layer=act_layer, norm_layer=norm_layer) # c5 -> c4
        self.dec_c3 = self._CombinationModule(channels[-2], channels[-3], act_layer=act_layer, norm_layer=norm_layer) # c4 -> c3
        self.dec_c2 = self._CombinationModule(channels[-3], channels[-4], act_layer=act_layer, norm_layer=norm_layer) # c3 -> c2

        self.conv_c4 = self._UpsampleConvBlock(in_channels=channels[-2], out_channels=channel, scale_factor=4, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_c3 = self._UpsampleConvBlock(in_channels=channels[-3], out_channels=channel, scale_factor=2, act_layer=act_layer, norm_layer=norm_layer)
        self.conv_c2 = nn.Conv2d(channels[-4], channel, kernel_size=1)
        
        # 最终的多尺度融合模块
        self.fusion = self._FusionModule(in_channels_per_input=channel, out_channels=channel, act_layer=act_layer, norm_layer=norm_layer)
        
        # 动态创建输出头
        self._create_output_heads(channel, act_layer, norm_layer)

    def _create_output_heads(self, final_channel: int, act_layer: nn.Module, norm_layer: nn.Module):
        """一个私有方法，用于创建所有预测头。"""
        # 定义不同头的卷积层输出通道数
        head_conv_dims = {
            'wh': 256,
            'hm': 256,
            'reg': 256,
        }
        
        for head, num_classes in self.heads.items():
            head_conv_dim = head_conv_dims.get(head, 256) # 获取该头的卷积维度
            
            if head == 'wh':
                # 'wh' 头使用较大的 7x7 卷积核
                fc = nn.Sequential(
                    nn.Conv2d(final_channel, head_conv_dim, kernel_size=7, padding=3, bias=True),
                    norm_layer(head_conv_dim),
                    act_layer(),
                    nn.Conv2d(head_conv_dim, num_classes, kernel_size=7, padding=3, bias=True)
                )
                self._initialize_weights(fc)
            else:
                # 'hm', 'reg' 等其他头使用 3x3 卷积核
                fc = nn.Sequential(
                    nn.Conv2d(final_channel, head_conv_dim, kernel_size=3, padding=1, bias=True),
                    norm_layer(head_conv_dim),
                    act_layer(),
                    nn.Conv2d(head_conv_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
                
                if 'hm' in head:
                    # 对热力图(hm)头的最终偏置进行特殊初始化，使其初始预测倾向于背景
                    fc[-1].bias.data.fill_(-2.19) 
                else:
                    self._initialize_weights(fc)

            # 将创建好的卷积头 fc 注册为模块的属性
            self.__setattr__(head, fc)

    def _initialize_weights(self, layers: nn.Module):
        """私有方法：初始化卷积层和归一化层的权重。"""
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    # 使用 Kaiming He 初始化卷积核权重
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                # 初始化归一化层的 gamma (weight) 为 1, beta (bias) 为 0
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # 3. 前向传播逻辑
    def forward(self, features: list) -> tuple:
        """
        前向传播。
        输入 `features` 是一个列表，包含来自骨干网络从低分辨率到高分辨率的特征图。
        例如：[c2, c3, c4, c5]
        """
        # x[-1] = c5 (最高层), x[-2] = c4, x[-3] = c3, x[-4] = c2 (最低层)
        # 逐步融合特征
        c4_features = self.dec_c4(features[-1], features[-2])
        c3_features = self.dec_c3(c4_features, features[-3])
        c2_features = self.dec_c2(c3_features, features[-4])

        # 对齐各层特征图的通道数，并进行上采样
        out_c4 = self.conv_c4(c4_features)
        out_c3 = self.conv_c3(c3_features)
        out_c2 = self.conv_c2(c2_features)

        # 最终融合
        final_features = self.fusion(out_c2, out_c3, out_c4)
        
        # 为每个头生成多尺度输出和最终融合输出
        outputs = {}
        # 为了与原始输出格式保持一致，我们创建多个字典
        # 如果不需要多尺度监督，可以简化这部分
        dec_dict_c2 = {}
        dec_dict_c3 = {}
        dec_dict_c4 = {}
        dec_dict_final = {}

        for head in self.heads:
            output_layer = self.__getattr__(head)
            dec_dict_c2[head] = output_layer(out_c2)
            dec_dict_c3[head] = output_layer(out_c3)
            dec_dict_c4[head] = output_layer(out_c4)
            dec_dict_final[head] = output_layer(final_features)

            # 对热力图应用 sigmoid 激活函数
            if 'hm' in head:
                dec_dict_c2[head] = torch.sigmoid(dec_dict_c2[head])
                dec_dict_c3[head] = torch.sigmoid(dec_dict_c3[head])
                dec_dict_c4[head] = torch.sigmoid(dec_dict_c4[head])
                dec_dict_final[head] = torch.sigmoid(dec_dict_final[head])

        return dec_dict_c2, dec_dict_c3, dec_dict_c4, dec_dict_final