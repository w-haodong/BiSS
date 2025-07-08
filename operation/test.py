import torch
import numpy as np
from models import spinal_net
import torch.nn.functional as F
import cv2
from operation import decoder
import os
from datasets.dataset import BaseDataset
from datasets import draw_points
import matplotlib
from operation import cobb_evaluate_base
import matplotlib.pyplot as plt
import scipy.io as sio

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = np.random.rand(3)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class Network(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes,
                 'reg': 2*args.num_classes,
                 'wh': 2*4,}

        self.model = spinal_net.SpineNet(heads=heads,
                                         down_ratio=args.down_ratio,
                                         device=self.device
                                         )
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K)
        self.dataset = {'spinal': BaseDataset}

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights_spinal from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict=False)
        return model

    def map_mask_to_image(self, mask, img, color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)


    def test(self, args, save):
        save_path =  args.work_dir+'/weights_'+args.dataset
        self.model = self.load_model(self.model, os.path.join(save_path, args.resume))
        self.model = self.model.to(self.device)
        self.model.eval()

        dataset_module = self.dataset[args.dataset]
        dsets = dataset_module(data_dir=args.data_dir,
                               phase='test',
                               input_h=args.input_h,
                               input_w=args.input_w,
                               down_ratio=args.down_ratio)

        data_loader = torch.utils.data.DataLoader(dsets,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        for cnt, data_dict in enumerate(data_loader):
            images = data_dict['images'][0]
            img_id = data_dict['img_id'][0]
            images = images.to('cuda')
            print('processing {}/{} image ... {}'.format(cnt, len(data_loader), img_id))
            with torch.no_grad():
                output,_,_,_ = self.model(images)
                hm = output['hm']
                wh = output['wh']
                reg = output['reg']

            torch.cuda.synchronize(self.device)
            pts2 = self.decoder.ctdet_decode(hm, wh, reg)   # 17, 11
            pts0 = pts2.copy()
            pts0[:,:10] *= args.down_ratio

            ori_image = dsets.load_image(dsets.img_ids.index(img_id), is_train = False)
            image_cobb_pr = ori_image.copy()
            image_cobb_gt = ori_image.copy()
            ori_image_regress = ori_image
            ori_image_points = ori_image_regress.copy()

            h,w,c = ori_image.shape
            pts0 = np.asarray(pts0, np.float32)
            pts0[:,0::2] = pts0[:,0::2]/args.input_w*w
            pts0[:,1::2] = pts0[:,1::2]/args.input_h*h
            sort_ind = np.argsort(pts0[:,1])
            pts0 = pts0[sort_ind]
            pr_landmarks = []
            for i, pt in enumerate(pts0):
                pr_landmarks.append(pt[2:4])
                pr_landmarks.append(pt[4:6])
                pr_landmarks.append(pt[6:8])
                pr_landmarks.append(pt[8:10])
            pr_landmarks = np.asarray(pr_landmarks, np.float32)  # [68, 2]
            
            cobb_angle1, cobb_angle2, cobb_angle3 = cobb_evaluate_base.cobb_angle_calc(pr_landmarks, image_cobb_pr)
            print('npr_ca1: {:.2f}, npr_ca2: {:.2f}, npr_ca3: {:.2f}'.format(cobb_angle1, cobb_angle2, cobb_angle3))

            ori_image_regress, ori_image_points = draw_points.draw_landmarks_regress_test(pts0,ori_image_regress,ori_image_points)

            # 创建目录（如果不存在），exist_ok参数避免目录存在时报错
            os.makedirs(args.work_dir+'/img_show', exist_ok=True)

            # 使用安全的路径拼接方式保存图像
            cv2.imwrite(os.path.join(args.work_dir+'/img_show', 'ca_pr_'+img_id), image_cobb_pr)
            self.save_heatmap(hm, os.path.join(args.work_dir+'/img_show', 'hm_'+img_id))

    def save_heatmap(self, hm_tensor, output_path="heatmap.png"):
        """
        保存热图张量为彩色图像文件
        参数:
            hm_tensor (Tensor): 形状为 (1, 1, H, W) 的CUDA张量
            output_path (str): 输出文件路径
        """
        # 1. 转换张量到CPU并提取数据
        #hm_tensor = self._nms(hm_tensor)
        hm_np = hm_tensor.detach().cpu().numpy()[0,0]  # 去除批次和通道维度
        
        # 2. 对数变换增强低值区域可见性
        eps = 1e-12  # 防止log(0)
        log_hm = np.log(hm_np + eps)
        
        # 3. 特殊归一化处理（针对极小数优化）
        min_val = np.min(log_hm)
        max_val = np.max(log_hm)
        normalized = (log_hm - min_val) / (max_val - min_val + 1e-8)
        uint8_hm = (normalized * 255).astype(np.uint8)
        
        # 4. 应用JET颜色映射
        color_hm = cv2.applyColorMap(uint8_hm, cv2.COLORMAP_JET)
        
        # 5. 保存结果
        cv2.imwrite(output_path, color_hm)
        print(f"Heatmap saved to {output_path}")

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep
