import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.bevformer.modules.decoder import CustomMSDeformableAttention
from .builder import build_fuser, FUSERS
from typing import List
from projects.mmdet3d_plugin.maptr.modules.base import BevEncode,BevEncode_v1,CLS_Head,CLS_HeadV2
from einops import rearrange
from projects.mmdet3d_plugin.maptr.dense_heads.maptr_head import normalize_2d_pts,denormalize_2d_pts
from mmseg.ops import resize
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
from .gru import ConvGRU,NewGRU,DeformConv2d
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mmdet.models.utils.transformer import inverse_sigmoid
@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=100,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 feat_down_sample_indice=-1,
                 modality='vision',
                 **kwargs):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

        self.feat_down_sample_indice = feat_down_sample_indice
        self.bev_h = bev_h
        self.bev_w = bev_w
        #self.merge = ConvGRU(256)
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        if self.use_attn_bev:
            self.level_embeds = nn.Parameter(torch.Tensor(
                self.num_feature_levels, self.embed_dims))
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims))

            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)  # TODO, this is a hack

    def init_weights(self):
        """Initialize the transformer weights."""
        if self.use_attn_bev:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for m in self.modules():
                if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                        or isinstance(m, CustomMSDeformableAttention):
                    try:
                        m.init_weight()
                    except AttributeError:
                        m.init_weights()
            normal_(self.level_embeds)
            normal_(self.cams_embeds)
            #xavier_init(self.reference_points, distribution='uniform', bias=0.)
            xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            pre_feat=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)#200*100 4 256
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return bev_embed

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        assert len(mlvl_feats) == 1, 'Currently we only support single level feat in LSS'
        # images = mlvl_feats[0]
        # img_metas = kwargs['img_metas']
        # bev_embed = self.encoder(images,img_metas)
        # bs, c, _,_ = bev_embed.shape
        # bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
        #
        # return bev_embed
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images, img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()

        return bev_embed,depth

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            pre_feat=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                pre_feat=pre_feat,
                **kwargs)
            depth=None
        else:
            bev_embed,depth = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h,bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev

        return bev_embed,depth

    def normalize(self,x):
        b,i,p = x.shape
        x_min,_ = torch.min(x,dim=-1)#4 50
        x_min = x_min.unsqueeze(2).repeat(1,1,p)
        x_max, _ = torch.max(x, dim=-1)#4 50
        x_max = x_max.unsqueeze(2).repeat(1, 1, p)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return x
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))

    def time_fuse(self,prev_bev,now_bev,bev_h,bev_w,**kwargs):
            bev = now_bev
            pre_ori = rearrange(prev_bev,' (h w) b c -> b c h w',h=bev_h,w =bev_w)
            b = prev_bev.shape[1]
            prev_bev = rearrange(prev_bev,' (h w) b c -> b c h w',h=bev_h,w =bev_w)

            now_rot = np.array( [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])  # 角度当前时刻的位姿
            delta_x = np.array([each['can_bus'][0] for each in kwargs['img_metas']])  # [bs] 前后两帧绝对距离的x分量-世界坐标系  正向为正值，反向为负值
            delta_y = np.array([each['can_bus'][1] for each in kwargs['img_metas']])  # [bs] 前后两帧绝对距离的y分量-世界坐标系  正向为正值，反向为负值
            delta_translation_global = torch.tensor(
                [delta_x, delta_y]).permute(1, 0).reshape(b, 2, 1)  # b 2 1

            curr_transform_matrix = torch.tensor([[np.cos(-now_rot), -np.sin(-now_rot)],
                                                  [np.sin(-now_rot), np.cos(-now_rot)]])  # (2, 2，bs) 世界坐标转换为当前帧自车坐标
            curr_transform_matrix = curr_transform_matrix.permute(2, 0, 1)  # b 2 2
            delta_translation_curr = torch.matmul(
                curr_transform_matrix,
                delta_translation_global).squeeze(-1)  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
            delta_translation_curr = delta_translation_curr // 0.3 # (bs, 2) = (bs, 2) / (1, 2) (格子数)
            delta_translation_curr = delta_translation_curr.round().tolist()

            rot = np.array([each['can_bus'][-1] for each in kwargs['img_metas']])  # 两帧相差角度，4

            map = []
            for i in range(b):
                tran = [-delta_translation_curr[i][0], delta_translation_curr[i][1]]
                pre_map_temp = prev_bev[i]
                pre_warp = affine(pre_map_temp,  # 必须是(c, h, w)形式的tensor
                                  angle=-rot[i],
                                  # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                  translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                  scale=1,  # 浮点数，中心缩放尺度
                                  shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                  interpolation=InterpolationMode.NEAREST,  # 二维线性差值，默认是最近邻差值
                                  fill=[0.0])

                pre_warp = torch.tensor(pre_warp)
                map.append(pre_warp)
            p = torch.stack(map).cuda()#b 256 200 100



            now_bev = rearrange(now_bev, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
            # # 1add
            # #bev_embed = prev_bev+now_bev
            # # bev_embed = self.merge(bev_embed)
            # # 2contact
            # #bev_embed = torch.cat((now_bev,prev_bev),dim=1)
            # #bev_embed = self.merge(bev_embed)
            # # 3gru
            bev_embed = self.merge(prev_bev,now_bev)
            # # 4newgru
            #bev_embed = self.merge(p,now_bev)
            #
            bev_embed = rearrange(bev_embed, 'b c h w -> b (h w)  c')
            show  = False
            if show:
                prev_bev_vis = rearrange(p,'b c h w -> b h w c')
                prev_bev_vis = torch.sum(prev_bev_vis,dim=3).cpu().numpy()
                now_bev_vis = rearrange(now_bev, ' b c h w -> b h w c')
                now_bev_vis = torch.sum(now_bev_vis, dim=3).cpu().numpy()
                now_bev_embed = rearrange(bev_embed, 'b (h w) c -> b h w c',h=self.bev_h, w=self.bev_w)#rearrange(prev_bev,'b c h w -> b h w c')#
                now_bev_embed = torch.sum(now_bev_embed, dim=3).cpu().numpy()
                plt.figure(figsize=(3, 2), dpi=100)
                plt.axis('off')  # 去坐标轴
                plt.xticks([])  # 去 x 轴刻度
                plt.yticks([])  # 去 y 轴刻度
                plt.axis('off')
                plt.subplot(131)
                plt.imshow(prev_bev_vis[0, :,:],cmap='Blues')

                plt.subplot(132)
                plt.imshow(now_bev_vis[0, :, :], cmap='Blues')
                plt.subplot(133)
                plt.imshow(now_bev_embed[0, :, :], cmap='Blues')
                plt.show()
            return bev_embed

    def position_encoder(self,pos_tensor,d_model=256):
        scale = 2 * math.pi
        dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos



    def show(self,img):
        plt.figure()

        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(img,cmap='Blues')
        plt.show()
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                pre_feat=None,
                mask_gt=None,
                object_query_embeds_one2many=None,
                **kwargs):

        bev_embed,depth = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            pre_feat=pre_feat,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        prev_embed = prev_bev
        # if prev_bev is not None:
        #     bev_embed = self.time_fuse(prev_bev,bev_embed,bev_h,bev_w,**kwargs)
        # vis = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
        # vis = torch.sum(vis,dim=1)
        # self.show(vis[0].cpu().numpy())
        his_embed = rearrange(bev_embed, 'b hw c -> hw b c')
        bs = mlvl_feats[0].size(0)
        if object_query_embed is not None:
            query_pos, query = torch.split(
                object_query_embed, self.embed_dims, dim=1)  # 1000 256
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # 4 1000 256
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)  # 4 1000 2
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points
            query = query.permute(1, 0, 2)  # 1000 4 256
            query_pos = query_pos.permute(1, 0, 2)
            bev_embed = bev_embed.permute(1, 0, 2)
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            semantic_mask = None
            instance_mask = None

        inter_references_out = inter_references

        return his_embed, inter_states, init_reference_out, inter_references_out,semantic_mask,instance_mask,depth

@TRANSFORMER.register_module()
class MapTRPerceptionTransformerV2(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=100,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                     pred_z_flag=False,
                     gt_z_flag=False,
                 ),
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,num_vec=100,
                 **kwargs):
        super(MapTRPerceptionTransformerV2, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser)  # TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg = z_cfg
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice
        self.bev_h = bev_h
        self.bev_w = bev_w
        # self.seg_out = BevEncode(inC=256, outC=1, seg_out=True, instance_seg=True, embedded_dim=50,many_instance_seg=True, many_instance_dim=50*6)
        # self.instance_embedding = nn.Embedding(50 * 20, self.embed_dims)
        # self.instance_embedding_many = nn.Embedding(50*6 * 20, self.embed_dims)
        # self.pos_mlp = MLP(256, 256, 256, 2)
        #self.cls =  BevEncode(inC=256, outC=4, seg_out=True,instance_seg=True, embedded_dim=50)#CLS_Head(256,50)
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0)
        )

        self.cls_head_1 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False)
        self.cls_head_2 = nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(self.embed_dims, num_vec, kernel_size=1, padding=0))
        # self.cls_head_11 = nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False)
        # self.cls_head_21 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.embed_dims, 300, kernel_size=1, padding=0))
        # self.instance_embedding = nn.Embedding(50 * 20, self.embed_dims)
        # self.instance_embedding_many = nn.Embedding(300 * 20, self.embed_dims)
        self.pos_mlp = MLP(256,256,256,2)
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg['gt_z_flag'] else nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                            for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                            for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
                  np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
                  np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        ret_dict = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            mlvl_feats=mlvl_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return ret_dict

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        # import ipdb;ipdb.set_trace()
        # assert len(mlvl_feats) == 1, 'Currently we only use last single level feat in LSS'
        # import ipdb;ipdb.set_trace()
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images, img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            ret_dict = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        else:
            ret_dict = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index

    def position_encoder(self,pos_tensor,d_model=256):
        scale = 2 * math.pi
        dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (d_model // 2))
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos

    def get_point(self,instance_mask,fea):
        instance = instance_mask.sigmoid()
        b, c, h, w = instance.shape
        out = rearrange(instance, 'b c h w -> b c (h w)')
        _, point = out.topk(20, dim=-1, largest=True, sorted=True)  # 4 50 20, sorted=True
        fea_point = rearrange(point, 'b i p -> b (i p)')
        p_y = fea_point // w  # 4 1000
        p_x = fea_point - p_y * w
        p_y = p_y / h
        p_x = p_x / w
        reference_points_ins = torch.stack((p_x, p_y), dim=-1)  # 4 1000 2
        reference_points_ins = reference_points_ins.sigmoid()
        query_pos = self.position_encoder(reference_points_ins)
        query_pos = self.pos_mlp(query_pos)
        fea_point1 = fea_point.unsqueeze(1).repeat(1, self.embed_dims, 1)
        query_pos_ins = torch.gather(fea, -1, fea_point1)  # 4 256 1000
        query_pos_ins = query_pos_ins.permute(0, 2, 1)  # 4 1000 256
        query_pos_ins += query_pos
        return reference_points_ins,query_pos_ins

    def CLS_learaning(self,bev_embed):

        bev = bev_embed
        semantic_mask, instance_mask, fea,instance_mask_many = self.seg_out(bev)  # _,4 50 200 100,4 256 200 100
        fea = rearrange(fea, 'b c h w -> b c (h w)')  # 4 256 200*100
        reference_points_ins,query_pos = self.get_point(instance_mask,fea)
        reference_points_ins_many,query_pos_many = self.get_point(instance_mask_many,fea)



        return reference_points_ins,query_pos,reference_points_ins_many,query_pos_many, instance_mask, semantic_mask,instance_mask_many

        # fea = cls_head_1(bev_embed)
        # instance_mask =cls_head_2(fea)
        # instance = instance_mask.sigmoid()
        #
        # b, c, h, w = instance.shape
        # out = rearrange(instance, 'b c h w -> b c (h w)')
        # _, point = out.topk(20, dim=-1, largest=True)  # 4 1 1000 , sorted=True
        # point, _ = point.topk(20, dim=-1, largest=True, sorted=True)
        # fea = rearrange(fea, 'b c h w -> b c (h w)')  # 4 256 200*100
        # fea_point = rearrange(point, 'b i p -> b (i p)')
        # p_y = fea_point // w  # 4 1000
        # p_x = fea_point - p_y * w
        # p_y = p_y / h #- 0.5
        # p_x = p_x / w #- 0.5
        # reference_points_ins = torch.stack((p_x, p_y), dim=-1)  # 4 1000 2
        # reference_points_ins = reference_points_ins.sigmoid()
        #
        # query_pos = self.position_encoder(reference_points_ins)
        # query_pos = self.pos_mlp(query_pos)
        #
        # fea_point1 = fea_point.unsqueeze(1).repeat(1, self.embed_dims, 1)
        # query_pos_ins = torch.gather(fea, -1, fea_point1)  # 4 256 1000
        # query_pos_ins = query_pos_ins.permute(0, 2, 1)  # 4 1000 256
        # query_pos_ins += query_pos
        # return query_pos_ins, reference_points_ins, instance_mask

    def point_learning(self,bev_embed,reference_points):
        fea = self.cls_head_1(bev_embed)
        instance_mask = self.cls_head_2(fea)  # b 50 200 100
        x = reference_points[:, :, 0] * (self.bev_w-1)#
        y = reference_points[:, :, 1] * (self.bev_h-1)
        pos = y * self.bev_w + x

        fea = rearrange(fea, 'b c h w -> b c (h w)')  # 4 256 200*100
        fea_point1 = pos.unsqueeze(1).repeat(1, self.embed_dims, 1)
        query_pos_ins = torch.gather(fea, -1, fea_point1.long())  # 4 256 1000
        query_pos_ins = query_pos_ins.permute(0, 2, 1)  # 4 1000 256
        query_pos = self.position_encoder(reference_points)
        query_pos = self.pos_mlp(query_pos)
        query_pos_ins += query_pos
        return query_pos_ins,instance_mask
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):

        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bev_embed = ouput_dic['bev']
        depth = ouput_dic['depth']
        bs = mlvl_feats[0].size(0)
        if object_query_embed is not  None:
            query_pos, query = torch.split(
                object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)#4 xx 256
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)#4 1000 2
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points
            # if query_pos.shape[1] ==300*20:
            #     query_pos_ins, reference_points_ins,instance_mask,semantic_mask = self.CLS_learaning(bev_embed,self.cls_head_1,self.cls_head_2)
            #     query_ins = self.instance_embedding.weight  # 50 256
            #     query_ins = query_ins.unsqueeze(0).expand(bs, -1, -1)  # 4 50 256
            #
            #     query_pos = torch.cat((query_pos_ins,query_pos),dim=1)
            #     query = torch.cat((query_ins,query),dim=1)
            #     reference_points = torch.cat((reference_points_ins,reference_points),dim=1) #4 1000+6000 2
            #     init_reference_out = reference_points
            # else:
            #     instance_mask=None
            #     semantic_mask=None
            if True:
                bev_embeds = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
                semantic_mask = self.seg_head(bev_embeds)
                query_pos_ins,  instance_mask = self.point_learning(bev_embeds,reference_points)
                query_pos= query_pos+query_pos_ins
            else:
                bev_embeds = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
                semantic_mask = self.seg_head(bev_embeds)
                instance_mask = None


        else:
            bev_embeds = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
            # semantic_mask = self.seg_head(bev_embeds)
            # query_pos_ins, reference_points_ins,instance_mask = self.CLS_learaning(bev_embeds,self.cls_head_1,self.cls_head_2,semantic_mask)
            reference_points_ins, query_pos, reference_points_ins_many, query_pos_many, instance_mask, semantic_mask, instance_mask_many =self.CLS_learaning(bev_embeds)
            query_ins = self.instance_embedding.weight  # 50 256
            query_ins = query_ins.unsqueeze(0).expand(bs, -1, -1)  # 4 50 256
            query_ins_many = self.instance_embedding_many.weight  # 50 256
            query_ins_many = query_ins_many.unsqueeze(0).expand(bs, -1, -1)  # 4 50 256
            if self.training:
                # query_pos_ins_many, reference_points_ins_many,instance_mask_many = self.CLS_learaning(bev_embeds, self.cls_head_11,self.cls_head_21,semantic_mask)
                # #query_pos_ins_many =  query_pos_ins.repeat(1,6,1)
                # #reference_points_ins_many = reference_points_ins.repeat(1,6,1)

                query_pos = torch.cat((query_pos,query_pos_many),dim=1)#query_pos_ins#
                query = torch.cat((query_ins,query_ins_many),dim=1)#query_ins
                reference_points = torch.cat((reference_points_ins,reference_points_ins_many),dim=1)#reference_points_ins  # 4 1000+6000 2
                instance_mask = torch.cat((instance_mask,instance_mask_many),dim=1)
            else:
                query_pos = query_pos#
                query = query_ins
                reference_points =  reference_points_ins  # 4 1000+6000 2

            init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        feat_flatten, feat_spatial_shapes, feat_level_start_index \
            = self.format_feats(mlvl_feats)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            mlvl_feats=mlvl_feats,
            feat_flatten=feat_flatten,
            feat_spatial_shapes=feat_spatial_shapes,
            feat_level_start_index=feat_level_start_index,
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, depth, inter_states, init_reference_out, inter_references_out,instance_mask,semantic_mask


@TRANSFORMER.register_module()
class MapTRPerceptionTransformer_v1(BaseModule):


    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 **kwargs):
        super(MapTRPerceptionTransformer_v1, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser)  # TODO
        self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Linear(self.embed_dims, 2) # TODO, this is a hack
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
        self.seg_out = BevEncode(inC=256, outC=4, seg_out=True, instance_seg=True, embedded_dim=50)
        self.instance_embedding = nn.Embedding(50 * 20, self.embed_dims)


    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            pre_feat=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # # obtain rotation angle and shift with ego motion
        # delta_x = np.array([each['can_bus'][0]
        #                     for each in kwargs['img_metas']])
        # delta_y = np.array([each['can_bus'][1]
        #                     for each in kwargs['img_metas']])
        # ego_angle = np.array(
        #     [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        # grid_length_y = grid_length[0]
        # grid_length_x = grid_length[1]
        # translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        # translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        # bev_angle = ego_angle - translation_angle
        # shift_y = translation_length * \
        #           np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        # shift_x = translation_length * \
        #           np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        # shift_y = shift_y * self.use_shift
        # shift_x = shift_x * self.use_shift
        # shift = bev_queries.new_tensor(
        #     [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy
        #
        # if prev_bev is not None:
        #     if prev_bev.shape[1] == bev_h * bev_w:
        #         prev_bev = prev_bev.permute(1, 0, 2)  # 200*100 4 256
        #     if self.rotate_prev_bev:
        #         for i in range(bs):
        #             # num_prev_bev = prev_bev.size(1)
        #             rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
        #             tmp_prev_bev = prev_bev[:, i].reshape(
        #                 bev_h, bev_w, -1).permute(2, 0, 1)
        #             tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
        #                                   center=self.rotate_center)
        #             tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
        #                 bev_h * bev_w, 1, -1)
        #             prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        if pre_feat is not None:
            feat_flatten_pre = []
            spatial_shapes_pre = []
            for lvl, feat in enumerate(pre_feat):
                bs, num_cam, c, h, w = feat.shape
                spatial_shape = (h, w)
                feat = feat.flatten(3).permute(1, 0, 3, 2)
                if self.use_cams_embeds:
                    feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
                feat = feat + self.level_embeds[None,
                              None, lvl:lvl + 1, :].to(feat.dtype)
                spatial_shapes_pre.append(spatial_shape)
                feat_flatten_pre.append(feat)

            feat_flatten_pre = torch.cat(feat_flatten_pre, 2)
            spatial_shapes_pre = torch.as_tensor(
                spatial_shapes_pre, dtype=torch.long, device=bev_pos.device)
            level_start_index_pre = torch.cat((spatial_shapes_pre.new_zeros(
                (1,)), spatial_shapes_pre.prod(1).cumsum(0)[:-1]))

            feat_flatten_pre = feat_flatten_pre.permute(
                0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
            flag = False
        else:
            spatial_shapes_pre = spatial_shapes
            level_start_index_pre = level_start_index
            feat_flatten_pre = feat_flatten
            flag = True
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            spatial_shapes_pre=spatial_shapes_pre,
            level_start_index_pre=level_start_index_pre,
            feat_pre=feat_flatten_pre,
            flag=flag,
            **kwargs
        )

        return bev_embed

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        assert len(mlvl_feats) == 1, 'Currently we only support single level feat in LSS'
        images = mlvl_feats[0]
        img_metas = kwargs['img_metas']
        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()

        return bev_embed

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            pre_feat=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                pre_feat=pre_feat,
                **kwargs)
        else:
            bev_embed = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev

        return bev_embed

    def normalize(self, x):
        b, i, p = x.shape
        x_min, _ = torch.min(x, dim=-1)  # 4 50
        x_min = x_min.unsqueeze(2).repeat(1, 1, p)
        x_max, _ = torch.max(x, dim=-1)  # 4 50
        x_max = x_max.unsqueeze(2).repeat(1, 1, p)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return x

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))

    def pre_instance(self, bev_embed, img_metas):
        with torch.no_grad():
            b = bev_embed.shape[1]
            bev = rearrange(bev_embed, '(h w) b  c -> b c h w', h=self.bev_h, w=self.bev_w)
            semantic_mask, instance_mask, fea = self.seg_out(bev)  # _,4 50 200 100,4 256 200 100
            now_rot = np.array([each['can_bus'][-2] / np.pi * 180 for each in img_metas])  # 角度当前时刻的位姿

            delta_x = np.array([each['can_bus'][0] for each in img_metas])  # [bs] 前后两帧绝对距离的x分量-世界坐标系  正向为正值，反向为负值
            delta_y = np.array([each['can_bus'][1] for each in img_metas])  # [bs] 前后两帧绝对距离的y分量-世界坐标系  正向为正值，反向为负值
            delta_translation_global = torch.tensor(
                [delta_x, delta_y]).permute(1, 0).reshape(b, 2, 1)  # b 2 1

            curr_transform_matrix = torch.tensor([[np.cos(-now_rot), -np.sin(-now_rot)],
                                                  [np.sin(-now_rot), np.cos(-now_rot)]])  # (2, 2，bs) 世界坐标转换为当前帧自车坐标
            curr_transform_matrix = curr_transform_matrix.permute(2, 0, 1)  # b 2 2
            delta_translation_curr = torch.matmul(
                curr_transform_matrix,
                delta_translation_global).squeeze(-1)  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
            delta_translation_curr = delta_translation_curr // 0.3  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
            delta_translation_curr = delta_translation_curr.round().tolist()

            rot = np.array([each['can_bus'][-1] for each in img_metas])  # 两帧相差角度，4

            prev_instance = []
            pre_feas = []
            for i in range(b):
                tran = [-delta_translation_curr[i][1], -delta_translation_curr[i][0]]
                pre_map_temp = instance_mask[i]
                pre_warp = affine(pre_map_temp,  # 必须是(c, h, w)形式的tensor
                                  angle=360 - rot[i],
                                  # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                  translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                  scale=1,  # 浮点数，中心缩放尺度
                                  shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                  interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                                  fill=[0.0])
                pre_warp = torch.tensor(pre_warp)
                prev_instance.append(pre_warp)

                pre_fea = affine(fea[i],  # 必须是(c, h, w)形式的tensor
                                 angle=-rot[i],
                                 # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                                 translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                                 scale=1,  # 浮点数，中心缩放尺度
                                 shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                                 interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                                 fill=[0.0])
                pre_fea = torch.tensor(pre_fea)
                pre_feas.append(pre_fea)
            prev_instance = torch.stack(prev_instance).cuda()  # b
            prev_feas = torch.stack(pre_feas).cuda()  # b
            return prev_instance, prev_feas

    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                pre_feat=None,
                **kwargs):


        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            pre_feat=pre_feat,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        if object_query_embed is not None:
            query_pos, query = torch.split(
                object_query_embed, self.embed_dims, dim=1)  # 1000 256
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # 4 1000 256
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)  # 4 1000 2
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points
            query = query.permute(1, 0, 2)  # 1000 4 256
            query_pos = query_pos.permute(1, 0, 2)
            bev_embed = bev_embed.permute(1, 0, 2)
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            semantic_mask = None
            instance_mask = None
        else:
            bev = rearrange(bev_embed, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w)
            semantic_mask, instance_mask, fea = self.seg_out(bev)  # _,4 50 200 100,4 256 200 100
            b, c, h, w = instance_mask.shape
            instance = F.relu(instance_mask)  # 4 50 200 100
            if False and prev_embed is not None:
                pre_instance, pre_fea = self.pre_instance(prev_embed, kwargs['img_metas'])
                pre_instance = F.relu(pre_instance)
                pre_instance = rearrange(pre_instance, 'b c h w -> b c (h w)')
                _, pre_point = pre_instance.topk(1, dim=-1, largest=True, sorted=True)  # 4 50 1
                now_instance = rearrange(instance, 'b c h w -> b c (h w)')
                _, now_point = now_instance.topk(1, dim=-1, largest=True, sorted=True)  # 4 50 1
                dist = torch.cdist(pre_point.float(), now_point.float())  # 4 50 50
                _, match = torch.min(dist, dim=1)  # 4 50
                pre_match_instance = torch.gather(pre_instance, 1, match.unsqueeze(2).repeat(1, 1, 20000))
                merge_instance = pre_match_instance + now_instance  ##4 50 200000
                _, point = merge_instance.topk(20, dim=-1, largest=True, sorted=True)  # 4 50 20
                fea = pre_fea + fea
            else:
                out = rearrange(instance, 'b c h w -> b c (h w)')
                _, point = out.topk(20, dim=-1, largest=True, sorted=True)  # 4 50 20

            # instance = rearrange(instance, 'b c h w -> b c (h w)')
            # _,center = torch.max(instance,dim=-1)#4 50
            # c_y = center // w#4 50
            # c_x = center - c_y * w

            # center = torch.stack((c_y,c_x),dim=-1)#4 50 2
            # center = center.unsqueeze(2).repeat(1,1,20,1)
            # center[:, :, 0] = center[:, :, 0] / h
            # center[:, :, 1] = center[:, :, 1] / w

            # instance_point = rearrange(instance_mask,'b c h w -> b c (h w)')
            # query_pos = self.query_pos(instance_point)#4 50 20*256
            # query_pos = rearrange(query_pos,'b c (p d) -> b (c p) d',p=20,d=self.embed_dims)#4 1000 256
            # relative_reference_points = self.reference_points(query_pos)#4 1000 2
            # relative_reference_points = rearrange(relative_reference_points,'b p (n c) -> b p n c',n=20,c=2)#4 50 20 2

            # center = rearrange(center,'b c p i -> b (c p) i')
            # reference_points = center + relative_reference_points#4 1000 2
            # reference_points = rearrange(reference_points,'b i p o -> b (i p) o')#4 1000 2
            # reference_points = reference_points.sigmoid()
            # query_pos = query_pos.permute(1,0,2)  # 1000 4 256

            fea = rearrange(fea, 'b c h w -> b c (h w)')  # 4 256 200*100
            # sample_point[:,:, 0] = sample_point[:,:, 0] * h
            # sample_point[:,:, 1] = sample_point[:,:, 1] *w
            # sample_point = sample_point.long()#4 1000 2
            # id = torch.zeros((b,50*20),dtype=torch.int64).cuda()#4 1000
            # id = id + sample_point[:,:,1] + sample_point[:,:,0] * w
            # id = torch.clamp(id,min=0,max=(w-1)*(h-1))
            # id = id.unsqueeze(1).repeat(1,self.embed_dims,1)
            fea_point = rearrange(point, 'b i p -> b (i p)')
            p_y = fea_point // w  # 4 1000
            p_x = fea_point - p_y * w
            p_y = p_y / h
            p_x = p_x / h
            reference_points = torch.stack((p_y, p_x), dim=-1)  # 4 1000 2
            reference_points = reference_points.sigmoid()
            fea_point = fea_point.unsqueeze(1).repeat(1, self.embed_dims, 1)
            query_pos = torch.gather(fea, -1, fea_point)  # 4 256 1000
            query_pos = query_pos.permute(2, 0, 1)  # 1000 4 256

            query = self.instance_embedding.weight  # 50 256
            query = query.unsqueeze(0).expand(bs, -1, -1)
            query = query.permute(1, 0, 2)  # 1000 4 256

            init_reference_out = reference_points  # 4 1000 2
            bev_embed = bev_embed.permute(1, 0, 2)
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out, semantic_mask, instance_mask

@TRANSFORMER.register_module()
class MapTRPerceptionTransformerV3(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,
                 **kwargs):
        super(MapTRPerceptionTransformerV3, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg=z_cfg
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg['gt_z_flag'] \
                            else nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def attn_bev_encode(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        ret_dict = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            mlvl_feats=mlvl_feats,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        return ret_dict

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        # import ipdb;ipdb.set_trace()
        # assert len(mlvl_feats) == 1, 'Currently we only use last single level feat in LSS'
        # import ipdb;ipdb.set_trace()
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images,img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        bs, c, _,_ = bev_embed.shape
        bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        if self.use_attn_bev:
            ret_dict = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        else:
            ret_dict = self.lss_bev_encode(
                mlvl_feats,
                prev_bev=prev_bev,
                **kwargs)
            bev_embed = ret_dict['bev']
            depth = ret_dict['depth']
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h,bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bev_embed = ouput_dic['bev']
        depth = ouput_dic['depth']
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        feat_flatten, feat_spatial_shapes, feat_level_start_index \
            = self.format_feats(mlvl_feats)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            mlvl_feats=mlvl_feats,
            feat_flatten=feat_flatten,
            feat_spatial_shapes=feat_spatial_shapes,
            feat_level_start_index=feat_level_start_index,
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, depth, inter_states, init_reference_out, inter_references_out

@TRANSFORMER.register_module()
class MapTRPerceptionTransformerV4(BaseModule):

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,
                 **kwargs):
        super(MapTRPerceptionTransformerV4, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg=z_cfg
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""

        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg['gt_z_flag'] \
                            else nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

        xavier_init(self.reference_points, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)

    def get_bev_features(self, x, img_metas):
        rots = []
        trans = []
        intrins = []
        post_rots, post_trans = [],[]
        for img_meta in img_metas:
            rots.append(img_meta['R'])
            trans.append(img_meta['T'])
            intrins.append(img_meta['camera_intrinsics'])
            post_trans.append(img_meta['post_tran'])
            post_rots.append(img_meta['post_rot'])
        rots = torch.tensor(rots).cuda()
        trans = torch.tensor(trans).cuda()
        intrins = torch.tensor(intrins).cuda()
        post_rots = torch.stack(post_rots).cuda()
        post_trans =  torch.stack(post_trans).cuda()
        bev_embed,semantic_map = self.encoder(x, rots,trans,  intrins, post_rots, post_trans,)
        ret_dict = dict(
            bev=bev_embed,
            depth=None,
            semantci_mask=semantic_map,
        )
        return ret_dict


    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                img,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        img_metas = kwargs['img_metas']

        ouput_dic = self.get_bev_features(img,img_metas )  # bev_embed shape: bs, bev_h*bev_w, embed_dims
        bev_embed = ouput_dic['bev']
        bev_embed = rearrange(bev_embed,'b c h w-> b (h w) c')
        map = ouput_dic['semantci_mask']

        bs = img.size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),

            **kwargs)

        inter_references_out = inter_references

        return bev_embed, map, inter_states, init_reference_out, inter_references_out


@TRANSFORMER.register_module()
class MapTRPerceptionTransformerV5(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 **kwargs):
        super(MapTRPerceptionTransformerV5, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) # TODO, this is a hack
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'), out_fp32=True)
    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )
        if lidar_feat is not None:
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h,bev_w), mode='bicubic', align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0,2,1).contiguous()
            bev_embed = fused_bev

        return bev_embed
    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):


        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out