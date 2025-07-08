import os
import torch.utils.data as data
import cv2
from scipy.io import loadmat
import numpy as np
import torch
from .draw_gaussian import *
from operation import transform
import math
import matplotlib.pyplot as plt


def rearrange_pts(pts):
    boxes = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
    return np.asarray(boxes, np.float32)


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = 68
        self.img_dir = os.path.join(data_dir, 'data', self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))

    def load_image(self, index, is_train = True):
        image = cv2.imread(os.path.join(self.img_dir, self.img_ids[index]))
        if is_train:
            enhanced_image = clahe_enhance(image)  # 使用原始输入图像
            image = unsharp_mask(enhanced_image)   # 需提前定义unsharp_mask函数
        return image

    def load_gt_pts(self, annopath):
        pts = loadmat(annopath)['p2']   # num x 2 (x,y)
        pts = rearrange_pts(pts)
        return pts

    def load_annoFolder(self, img_id):
        return os.path.join(self.data_dir, 'labels', self.phase, img_id+'.mat')

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        pts = self.load_gt_pts(annoFolder)
        return pts

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = self.load_image(index)
        ori_imgH = image.shape[0]
        ori_imgW = image.shape[1]
        if self.phase == 'test':
            images = processing_test(image=image, input_h=self.input_h, input_w=self.input_w)
            return {'images': images, 'img_id': img_id}
        else:
            aug_label = False
            if self.phase == 'train':
                aug_label = True
            pts = self.load_annotation(index)   # num_obj x h x w
            out_image, pts_2 = processing_train(image=image,
                                                         pts=pts,
                                                         image_h=self.input_h,
                                                         image_w=self.input_w,
                                                         down_ratio=self.down_ratio,
                                                         aug_label=aug_label,
                                                         img_id=img_id)

            data_dict = generate_ground_truth(image=out_image,
                                                       pts_2=pts_2,
                                                       image_h=self.input_h//self.down_ratio,
                                                       image_w=self.input_w//self.down_ratio,
                                                       img_id=img_id,
                                                       ori_imgH = ori_imgH,
                                                       ori_imgW = ori_imgW,
                                                       ori_pts = pts)
            return data_dict

    def __len__(self):
        return len(self.img_ids)

def clahe_enhance(image, clip_limit=3.0, grid_size=(16,8)):
    """适配脊柱X-Ray的CLAHE增强"""
    # 强制输入为uint8类型
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 处理三通道图像
    if len(image.shape) == 3:
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    else:  # 灰度图像
        return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size).apply(image)

def unsharp_mask(image, sigma=1.5, strength=1.2):
    """改进的非锐化掩模"""
    # 确保输入为uint8
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # 处理三通道图像
    if len(image.shape) == 3:
        sharpened = np.zeros_like(image)
        for c in range(3):  # 分通道处理
            blurred = cv2.GaussianBlur(image[:,:,c], (0,0), sigma)
            sharpened[:,:,c] = cv2.addWeighted(image[:,:,c], 1.0 + strength, 
                                              blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    else:  # 灰度图像
        blurred = cv2.GaussianBlur(image, (0,0), sigma)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
def processing_test(image, input_h, input_w):
    image = cv2.resize(image, (input_w, input_h))
    out_image = image.astype(np.float32) / 255.
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        cv2.putText(out_image, '{}'.format(i+1), (int(pts[i, 0]), int(pts[i, 1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0),1,1)
    for i,j in zip([0,1,2,3], [1,2,3,0]):
        cv2.line(out_image,
                 (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[j, 0]), int(pts[j, 1])),
                 color=colors[i], thickness=1, lineType=1)
    return out_image


def rearrange_pts(pts):
    # rearrange left right sequence
    boxes = []
    centers = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k:k+4,:]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:,1])
        y_inds_r = np.argsort(pt_r[:,1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
        centers.append(np.mean(pts_4, axis=0))
    bboxes = np.asarray(boxes, np.float32)
    # rearrange top to bottom sequence
    centers = np.asarray(centers, np.float32)
    sort_tb = np.argsort(centers[:,1])
    new_bboxes = []
    for sort_i in sort_tb:
        new_bboxes.append(bboxes[4*sort_i, :])
        new_bboxes.append(bboxes[4*sort_i+1, :])
        new_bboxes.append(bboxes[4*sort_i+2, :])
        new_bboxes.append(bboxes[4*sort_i+3, :])
    new_bboxes = np.asarray(new_bboxes, np.float32)
    return new_bboxes


# --- 主要的 Ground Truth 生成函数 ---
def generate_ground_truth(image,
                          pts_2,        # 关键点坐标 (在输出特征图尺度上)
                          image_h,      # 特征图高度
                          image_w,      # 特征图宽度
                          img_id,       # 图像标识符
                          ori_imgH,     # 原始图像高度
                          ori_imgW,     # 原始图像宽度
                          ori_pts      # 原始关键点坐标
                          ): 
    """
    生成 Ground Truth，包括热力图、偏移量、'wh'属性和 PAFs。

    Args:
        image: 输入图像张量/数组。
        pts_2: 输出特征图尺度上的关键点坐标 (Num_Keypoints * 4, 2 or Num_Keypoints, 2).
        image_h: 输出特征图的高度。
        image_w: 输出特征图的宽度。
        img_id: 图像标识符。
        ori_imgH: 原始图像高度。
        ori_imgW: 原始图像宽度。
        ori_pts: 原始关键点坐标。

    Returns:
        dict: 包含所有 Ground Truth 张量的字典。
    """
    num_vertebrae = 17 # 假设有 17 个椎骨关键点 (例如 T1-L5 的中心)
    output_h = image_h
    output_w = image_w

    # 初始化 GT 张量
    hm = np.zeros((1, output_h, output_w), dtype=np.float32)       # 热力图
    wh = np.zeros((num_vertebrae, 8), dtype=np.float32)           # 'wh' 属性
    reg = np.zeros((num_vertebrae, 2), dtype=np.float32)          # 亚像素偏移量
    ind = np.zeros((num_vertebrae), dtype=np.int64)               # 展平索引
    reg_mask = np.zeros((num_vertebrae), dtype=np.uint8)          # 有效关键点掩码
    centers_scaled = np.zeros((num_vertebrae, 2), dtype=np.float32) # 存储缩放后的中心点

    # --- 1. 处理关键点 (生成 hm, reg, wh, mask) ---
    is_pts_4_corners = (pts_2.shape[0] == num_vertebrae * 4) # 判断输入是角点还是中心点
    if is_pts_4_corners and pts_2.shape[0] != num_vertebrae * 4:
         print(f'注意!! 图像 {img_id} 的 pts_2 点数 {pts_2.shape[0]} 不等于预期的 {num_vertebrae * 4}!!! ')
         # Handle error
    elif not is_pts_4_corners and pts_2.shape[0] != num_vertebrae:
         print(f'注意!! 图像 {img_id} 的 pts_2 点数 {pts_2.shape[0]} 不等于预期的 {num_vertebrae}!!! ')
         # Handle error

    for k in range(num_vertebrae):
        if is_pts_4_corners:
            pts_k = pts_2[4*k : 4*k+4, :] # 获取角点
            # 检查角点坐标有效性
            if np.all(np.abs(pts_k) < 1e-6) or np.any(pts_k < 0) or \
               np.any(pts_k[:, 0] >= output_w) or np.any(pts_k[:, 1] >= output_h):
                continue # 跳过无效点
            cen_x, cen_y = np.mean(pts_k, axis=0) # 计算中心点
            ct = np.asarray([cen_x, cen_y], dtype=np.float32)
             # wh 计算 (到角点偏移)
            for i in range(4):
                wh[k, 2*i : 2*i+2] = ct - pts_k[i, :]
        else:
            # 直接使用中心点坐标
            ct = pts_2[k, :]
            # 检查中心点坐标有效性
            if np.any(ct < 0) or ct[0] >= output_w or ct[1] >= output_h:
                continue # 跳过无效点
            pts_k = ct # 方便后续计算，虽然没有角点信息了
            # 注意: 如果没有角点，'wh' 的GT如何定义？这里保持全零或需要其他定义
            # wh[k, :] = ... # 根据你的 wh 定义来填充

        # 计算整数坐标和半径 (逻辑同前，但基于中心点)
        ct_int = np.clip(ct, [0, 0], [output_w - 1, output_h - 1]).astype(np.int32)
        # 如果有角点，可以用bbox估计半径；如果没有，可能需要默认半径或基于其他信息
        if is_pts_4_corners:
             min_coords = np.min(pts_k, axis=0); max_coords = np.max(pts_k, axis=0)
             bbox_h_est = max(1, max_coords[1] - min_coords[1])
             bbox_w_est = max(1, max_coords[0] - min_coords[0])
             radius = gaussian_radius((math.ceil(bbox_h_est), math.ceil(bbox_w_est)))
        else:
             radius = 3 # 示例：使用默认半径，需要调整
        radius = max(0, int(radius))

        # 绘制高斯热图
        draw_umich_gaussian(hm[0, :, :], ct_int, radius=radius)

        # 存储 GT 值
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        centers_scaled[k] = ct


    # --- 3. 整理最终的 Ground Truth 字典 ---
    ret = {'input': image,           # 输入图像
           'hm': hm,                 # 热图 (1, H, W)
           'ind': ind,               # 展平索引 (Num_Vertebrae,)
           'reg': reg,               # 亚像素偏移 (Num_Vertebrae, 2)
           'wh': wh,                 # 'wh'属性 (Num_Vertebrae, 8)
           'reg_mask': reg_mask,     # 有效点掩码 (Num_Vertebrae,)
           # 可选: 保留原始信息
           'pts': pts_2,             # 缩放后的点 (取决于输入是角点还是中心点)
           'ori_imgH': np.array(ori_imgH), # 原始图像高
           'ori_imgW': np.array(ori_imgW), # 原始图像宽
           'ori_pts': np.array(ori_pts)    # 原始点
           }

    return ret

def processing_train(image, pts, image_h, image_w, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    h,w,c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    data_aug = {'train': transform.Compose([transform.ConvertImgFloat(),
                                            transform.PhotometricDistort(),
                                            transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                                            transform.Equalize(),
                                            transform.RandomMirror_w(),
                                            transform.Resize(h=image_h, w=image_w)]),
                'val': transform.Compose([transform.ConvertImgFloat(),
                                          transform.Resize(h=image_h, w=image_w)])}
    if aug_label:
        out_image, pts = data_aug['train'](image.copy(), pts)
    else:
        out_image, pts = data_aug['val'](image.copy(), pts)

    out_image = np.clip(out_image, a_min=0., a_max=255.)
    out_image = np.transpose(out_image / 255. - 0.5, (2,0,1))
    pts = rearrange_pts(pts)
    pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)

    return np.asarray(out_image, np.float32), pts2
