from .dec_net import DecNet
import torch.nn as nn# 引入 SS2Dv2
from . import enc_net
import numpy as np

class SpineNet(nn.Module):
    def __init__(self, heads, down_ratio, device):
        super(SpineNet, self).__init__()
        channels = [3, 64, 64, 128, 256, 512]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        # 保持 DecNet 实现不变
        self.enc_net= enc_net.biss_encoder(device = device, channels =  channels)
        # 根据 down_ratio 决定输入通道数
        self.dec_net = DecNet(heads, channels[self.l1], channels = channels)

    def forward(self, x):
        # 输入到编码模块
        x = self.enc_net(x)
        # 输入到解码模块
        dec_dict_c2, dec_dict_c3, dec_dict_c4, dec_dict = self.dec_net(x)
        return  dec_dict_c2, dec_dict_c3, dec_dict_c4, dec_dict