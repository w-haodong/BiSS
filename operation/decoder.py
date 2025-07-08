import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from scipy.interpolate import CubicSpline

SPINE_CONNECTION_INDICES = [ [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16] ]
NUM_CONNECTION_TYPES = len(SPINE_CONNECTION_INDICES)
NUM_KEYPOINTS = 17

# 在残差连接后添加可微截断
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0, max=1)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
        
class DecDecoder(object):
    def __init__(self, K, device = 'cuda:0'):
        self.K = K
        self.device = device

    def _topk(self, scores):
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float(); topk_xs = (topk_inds % width).int().float()
        topk_score_overall, topk_ind_overall = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_cat = (topk_ind_overall / self.K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind_overall).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind_overall).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind_overall).view(batch, self.K)
        return topk_score_overall, topk_inds, topk_cat, topk_ys, topk_xs


    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def ctdet_decode(self, heat, wh, reg):
        # output: num_obj x 7
        # 7: cenx, ceny, w, h, angle, score, cls
        batch, c, height, width = heat.size(); heat = self._nms(heat)
        scores, inds, cat, ys, xs = self._topk(heat); 
        scores = scores.view(batch, self.K, 1)
        reg = self._tranpose_and_gather_feat(reg, inds); 
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]; 
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        wh = self._tranpose_and_gather_feat(wh, inds); 
        wh = wh.view(batch, self.K, 8)
        tl_x = xs - wh[:,:,0:1]; tl_y = ys - wh[:,:,1:2]; 
        tr_x = xs - wh[:,:,2:3]; tr_y = ys - wh[:,:,3:4]
        bl_x = xs - wh[:,:,4:5]; bl_y = ys - wh[:,:,5:6]; 
        br_x = xs - wh[:,:,6:7]; br_y = ys - wh[:,:,7:8]
        cls = cat.view(batch, self.K, 1).float()
        pts = torch.cat([xs, ys, tl_x,tl_y, tr_x,tr_y, bl_x,bl_y, br_x,br_y, scores, cls], dim=2)
        if batch == 1: return pts.squeeze(0).data.cpu().numpy()
        else: return [p.data.cpu().numpy() for p in pts]

    def decode_centers(self, heat, reg):
        """ 关键代码修正：规范参数传递 """
        batch, c, height, width = heat.size()
        
        # 执行NMS和非极大值抑制
        heat = self._nms(heat)
        scores, inds, ys, xs = self._topk(heat)  # [B,K], [B,K], [B,K]
        
        # 修正维度处理
        reg = self._tranpose_and_gather_feat(reg, inds)  # [B,K,2]
        reg = reg.view(batch, self.K, 2)
        
        # 坐标解码
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]  # [B,K,1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        
        # 转换为归一化坐标
        img_size = torch.tensor([width, height], device=heat.device)
        centers = torch.cat([xs, ys], dim=2) / img_size  # [B,K,2]
        # 按y坐标排序（升序排列）
        sort_ind = torch.argsort(centers[..., 1], dim=1)  # [B, N]
        
        # 使用排序索引重组坐标张量
        centers = torch.gather(
            centers, 
            1, 
            sort_ind.unsqueeze(-1).expand(-1, -1, 2)
        )  # [B, N, 2]

        return centers
    
    def postprocess_pts(self, pts):
        # 输入形状: (K, 11)
        # 显式添加 Batch 维度
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)  # 形状变为 (1, K, 11)

        # 提取中心点和4个关键点
        centers = pts[:, :, 0:2]  # (1, K, 2)
        tl = pts[:, :, 2:4]       # 左上关键点
        tr = pts[:, :, 4:6]       # 右上关键点
        bl = pts[:, :, 6:8]       # 左下关键点
        br = pts[:, :, 8:10]      # 右下关键点

        # 合并为 (1, K, 10) → 中心点 + 4关键点
        keypoints = torch.cat([centers, tl, tr, bl, br], dim=2)

        # 移除 Batch 维度（可选）
        return keypoints.squeeze(0)  # 形状 (K, 10)
    
    @torch.no_grad()
    def decode_sequence_with_pafs(self,
                                 heat: torch.Tensor,
                                 reg: torch.Tensor,
                                 wh: torch.Tensor, 
                                 paf: torch.Tensor,
                                 score_thresh: float = 0.1,
                                 paf_score_th: float = 0.05,
                                 paf_sample_steps: int = 10,
                                 min_spine_kpts: int = 5):
        """
        使用热力图、偏移量、wh 和 PAFs 解码，组装成带有详细信息的有序脊柱序列。

        Args:
            heat (Tensor): 预测的热力图 [B, C, H, W]。
            reg (Tensor): 预测的偏移量回归 [B, 2, H, W]。
            wh (Tensor): 预测的 'wh' 属性 [B, 8, H, W]。
            paf (Tensor): 预测的部分亲和场 [B, Num_Connections*2, H, W]。
            # ... 其他参数 ...

        Returns:
            list: 包含批次中每张图像检测到的脊柱序列的列表。
                  每个脊柱是一个 numpy 数组，形状为 (Num_Detected_Kpts, 11)，
                  每行是 (cx, cy, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, score)。
                  坐标是特征图尺度上的。
        """
        batch_size, num_classes, height, width = heat.size()
        num_connections = paf.size(1) // 2
        num_wh_channels = wh.size(1) # 获取 wh 的通道数 (应该是 8)

        if num_connections != len(SPINE_CONNECTION_INDICES):
             raise ValueError("PAF channel mismatch")
        if num_wh_channels != 8: # 检查 wh 通道数
             print(f"警告: 输入 'wh' 的通道数 ({num_wh_channels}) 不是预期的 8。角点计算可能不正确。")


        # --- 步骤 1: 检测候选关键点 ---
        heat = self._nms(heat)
        scores_all, inds_all, cat_all, ys_all, xs_all = self._topk(heat)
        reg_all = self._tranpose_and_gather_feat(reg, inds_all)
        xs_all_refined = xs_all.view(batch_size, -1, 1) + reg_all[..., 0:1] # 精确 x 坐标
        ys_all_refined = ys_all.view(batch_size, -1, 1) + reg_all[..., 1:2] # 精确 y 坐标

        all_candidates = [[] for _ in range(batch_size)]
        candidate_coords_tensor = torch.cat([xs_all_refined, ys_all_refined], dim=-1) # [B, C*K, 2]
        for b in range(batch_size):
            for k in range(scores_all.size(1)):
                score = scores_all[b, k].item()
                if score < score_thresh: continue
                x = xs_all_refined[b, k, 0].item()
                y = ys_all_refined[b, k, 0].item()
                kpt_type = cat_all[b, k].item()
                cand_id = k # 使用原始索引 k 作为 ID
                # 存储精确坐标，方便后续插值 wh
                all_candidates[b].append({'x': x, 'y': y, 'score': score, 'id': cand_id, 'type': kpt_type})

        # --- 步骤 2 & 3: 计算连接得分并组装脊柱序列 ---
        final_spine_data_batch = [] # 存储带详细信息的脊柱数据
        for b in range(batch_size):
            candidates = all_candidates[b]
            if not candidates:
                final_spine_data_batch.append([])
                continue

            # --- 计算连接得分 ---
            connection_candidates = []
            for conn_idx, (kpt_type1_idx, kpt_type2_idx) in enumerate(SPINE_CONNECTION_INDICES):
                paf_ch_x, paf_ch_y = conn_idx * 2, conn_idx * 2 + 1
                for cand1 in candidates:
                    for cand2 in candidates:
                        if cand1['id'] == cand2['id']: continue
                        p1 = torch.tensor([cand1['x'], cand1['y']], device=self.device); p2 = torch.tensor([cand2['x'], cand2['y']], device=self.device)
                        vec = p2 - p1; dist = torch.linalg.norm(vec)
                        if dist < 1e-2: continue
                        unit_vec = vec / dist
                        samples_x = torch.linspace(p1[0], p2[0], steps=paf_sample_steps, device=self.device); samples_y = torch.linspace(p1[1], p2[1], steps=paf_sample_steps, device=self.device)
                        px_norm = samples_x.clamp(0, width - 1 - 1e-4); py_norm = samples_y.clamp(0, height - 1 - 1e-4)
                        px_grid = (px_norm / (width - 1)) * 2 - 1; py_grid = (py_norm / (height - 1)) * 2 - 1
                        grid = torch.stack([px_grid, py_grid], dim=-1).unsqueeze(0).unsqueeze(0) # [1, 1, N, 2]
                        paf_channels = paf[b, paf_ch_x : paf_ch_y + 1].unsqueeze(0) # [1, 2, H, W]
                        paf_vectors = F.grid_sample(paf_channels, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze() # [2, N]
                        scores_dot = torch.matmul(unit_vec.unsqueeze(0), paf_vectors) # [1, N]
                        score = scores_dot.mean()
                        if score > paf_score_th:
                             weighted_score = score * (cand1['score'] + cand2['score'])
                             connection_candidates.append({'id1': cand1['id'], 'id2': cand2['id'], 'score': score.item(), 'weighted_score': weighted_score.item(), 'conn_idx': conn_idx})

            # --- 组装脊柱序列 ---
            subsets = {}
            used_cand_ids = set()
            connection_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
            for conn in connection_candidates:
                id1, id2 = conn['id1'], conn['id2']; conn_idx = conn['conn_idx']
                found_extension = False
                if conn_idx > 0:
                    prev_conn_idx = conn_idx - 1
                    for end_id, data in list(subsets.items()):
                        if end_id == id1 and data['last_conn_idx'] == prev_conn_idx and id2 not in used_cand_ids:
                            new_kpts = data['kpts'] + [id2]; new_score = data['score'] + conn['weighted_score']
                            if id2 not in subsets or new_score > subsets[id2]['score']:
                                subsets[id2] = {'kpts': new_kpts, 'score': new_score, 'last_conn_idx': conn_idx}
                                used_cand_ids.add(id2); found_extension = True #; break # Simplified: allow multiple extensions start
                if not found_extension and conn_idx == 0:
                    if id1 not in used_cand_ids and id2 not in used_cand_ids:
                         if id2 not in subsets or conn['weighted_score'] > subsets[id2]['score']:
                             subsets[id2] = {'kpts': [id1, id2], 'score': conn['weighted_score'], 'last_conn_idx': conn_idx}
                             used_cand_ids.add(id1); used_cand_ids.add(id2)

            # --- 步骤 4: 过滤、整合wh信息并格式化 ---
            final_spines_data = []
            cand_map = {c['id']: c for c in candidates}

            # 准备用于插值采样的 grid (需要所有最终选定点的坐标)
            points_to_sample_wh = []
            spine_indices_map = {} # 记录哪些点属于哪个最终序列

            for spine_idx, (end_id, data) in enumerate(subsets.items()):
                if len(data['kpts']) >= min_spine_kpts:
                    valid_spine_points = []
                    for kpt_id in data['kpts']:
                        if kpt_id in cand_map:
                            cand = cand_map[kpt_id]
                            points_to_sample_wh.append([cand['x'], cand['y']])
                            valid_spine_points.append(cand) # 存储完整的候选点信息
                            # 记录这个点属于哪个最终spine的索引
                            if kpt_id not in spine_indices_map: spine_indices_map[kpt_id] = []
                            spine_indices_map[kpt_id].append(spine_idx) # 一个点可能属于多个最终序列？(如果组装逻辑允许多重扩展)

                    if valid_spine_points: # 如果序列有效
                         final_spines_data.append(valid_spine_points) # 存储每个spine的候选点信息列表

            # 如果没有有效的脊柱序列，直接返回空列表
            if not final_spines_data:
                 final_spine_data_batch.append([])
                 continue

            # --- 批量插值采样 wh ---
            # 将所有需要采样 wh 的点坐标收集起来
            points_tensor = torch.tensor(points_to_sample_wh, dtype=torch.float32, device=self.device) # Shape [N_total_pts, 2]
            if points_tensor.numel() > 0: # 只有当有点需要采样时才进行
                px_norm = points_tensor[:, 0].clamp(0, width - 1 - 1e-4)
                py_norm = points_tensor[:, 1].clamp(0, height - 1 - 1e-4)
                px_grid = (px_norm / (width - 1)) * 2 - 1
                py_grid = (py_norm / (height - 1)) * 2 - 1
                # Grid shape for grid_sample: [1, N, 1, 2]
                grid_wh = torch.stack([px_grid, py_grid], dim=-1).unsqueeze(0).unsqueeze(1)

                # 采样 wh 图 [B, C_wh, H, W] -> [1, C_wh, H, W] (取当前 batch item)
                sampled_wh = F.grid_sample(
                    wh[b].unsqueeze(0), # [1, C_wh, H, W]
                    grid_wh,            # [1, N_total_pts, 1, 2] ??? -> 应为 [1, 1, N, 2]
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ) # Output: [1, C_wh, 1, N_total_pts] -> squeeze -> [C_wh, N_total_pts]
                sampled_wh = sampled_wh.squeeze().permute(1, 0) # Shape [N_total_pts, C_wh]
            else:
                sampled_wh = torch.empty((0, num_wh_channels), device=self.device)

            # --- 将采样到的 wh 值分配回对应的脊柱点 ---
            wh_lookup = {} # { kpt_id: wh_vector }
            point_idx_counter = 0
            processed_kpt_ids = set() # 处理重复的点ID (如果一个点在多个spine中)
            for spine in final_spines_data:
                 for cand in spine:
                     kpt_id = cand['id']
                     # 只有当这个kpt_id还没被分配wh值时才分配
                     if kpt_id not in processed_kpt_ids:
                         wh_lookup[kpt_id] = sampled_wh[point_idx_counter].cpu().numpy()
                         processed_kpt_ids.add(kpt_id)
                         point_idx_counter += 1


            # --- 构建最终输出格式 ---
            output_spines = []
            for spine_points_info in final_spines_data: # 这是一个包含候选点字典的列表
                spine_output_rows = []
                valid_spine = True
                for cand in spine_points_info:
                    kpt_id = cand['id']
                    if kpt_id not in wh_lookup:
                        print(f"警告: 无法为 kpt_id {kpt_id} 找到对应的 wh 值。跳过此脊柱。")
                        valid_spine = False
                        break # 跳过这个脊柱

                    center_x = cand['x']
                    center_y = cand['y']
                    score = cand['score']
                    wh_vec = wh_lookup[kpt_id] # 获取8维wh向量

                    # 计算角点
                    tl_x = center_x - wh_vec[0]; tl_y = center_y - wh_vec[1]
                    tr_x = center_x - wh_vec[2]; tr_y = center_y - wh_vec[3]
                    bl_x = center_x - wh_vec[4]; bl_y = center_y - wh_vec[5]
                    br_x = center_x - wh_vec[6]; br_y = center_y - wh_vec[7]

                    # 组合成一行输出 (11列: cx, cy, tlxy, trxy, blxy, brxy, score)
                    row = [center_x, center_y, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, score]
                    spine_output_rows.append(row)

                if valid_spine and spine_output_rows: # 如果脊柱有效且包含点
                    output_spines.append(np.array(spine_output_rows, dtype=np.float32))

            final_spine_data_batch.append(output_spines)
        return final_spine_data_batch

    