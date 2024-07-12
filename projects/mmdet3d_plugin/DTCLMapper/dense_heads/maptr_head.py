import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from shapely.geometry import LineString, box,LinearRing
from shapely import affinity
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import L1Loss
from einops import rearrange
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

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


class MemoryBank_temp:
    """
    众所周知，就是一个memory bank，主要用来存取 tracklet，与 CL 中的 Memory Bank 有一定的区别
    """

    def __init__(self,
                 b=4):
        #self.tracklets = {0:[],1:[],2:[]}
        self.tracklets=[]
        for i in range(b):
            self.tracklets.append({0:[],1:[],2:[]})
        #self.tracklets = [{0:[],1:[],2:[]},{0:[],1:[],2:[]},{0:[],1:[],2:[]},{0:[],1:[],2:[]}]
        self.update_flag=False
    def clean(self,instance_id,b=0):
        self.tracklets[b][instance_id] = []
    def update(self, instance_id, emb,b=0,bbox=None):
        self.tracklets[b][instance_id].append({'emb':emb,'bbox':bbox})
        self.update_flag = True
    def get(self, instance_id,b=0):
        return self.tracklets[b][instance_id]
        # if len(self.tracklets[b][instance_id])>0:
        #     return self.tracklets[b][instance_id]
        # else:
        #     if pos:
        #         return [float('inf')]*256
        #     else:
        #         return [float('-inf')]*256

    def is_notempty(self,instance_id,b=0):
        if len(self.tracklets[b][instance_id])>0:
            return True
        else:
            return False


class MemoryBank:
    """
    众所周知，就是一个memory bank，主要用来存取 tracklet，与 CL 中的 Memory Bank 有一定的区别
    """

    def __init__(self,
                 maximum_cache=5):
        self.tracklets = {0:[],1:[],2:[]}

    def clean(self,instance_id,b=0):
        self.tracklets[instance_id] = []
    def update(self, instance_id, emb,b=0):
        self.tracklets[instance_id].append(emb)
        #self.tracklets[instance_id]=emb
    def get(self, instance_id,pos=True,b=0):

        if len(self.tracklets[instance_id])>0:
            return self.tracklets[instance_id]
        else:
            if pos:
                return [torch.tensor([float('inf')]*256)]
            else:
                return [torch.tensor([float('-inf')]*256)]

    def is_notempty(self,instance_id,b=0):
        if len(self.tracklets[instance_id])>0:
            return True
        else:
            return False

@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec=20,
                 num_vec_one2one=50,
                 num_vec_one2many=0,
                 k_one2many=0,
                 lambda_one2many=1,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 z_cfg=dict(
                     pred_z_flag=False,
                     gt_z_flag=False,
                 ),
                 loss_pts=dict(type='ChamferDistance', 
                             loss_src_weight=1.0, 
                             loss_dst_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_seg=dict(type='SegmentationLoss', class_weights=[1.0, 2.0, 2.0, 2.0], weight=5.0),
                 loss_ins=dict(type='DiscriminativeLoss', embed_dim=50, delta_v=0.5, delta_d=3.0),
                 #loss_ins=dict(type='SimpleLoss', pos_weight=2.13, loss_weight=1.0),
                 loss_tda=dict(type='TDALoss', loss_weight=0.0),
                 loss_geo=dict(
                     type='GeometricLoss',
                     loss_weight=0.005,
                     intra_loss_weight=1.0,
                     inter_loss_weight=1.0,
                     num_ins=50,
                     num_pts=20,
                     pc_range=[-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                     loss_type='l1',
                 ),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.z_cfg = z_cfg
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern

        num_vec = num_vec_one2one + num_vec_one2many
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        
        
        super(MapTRHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_dir = build_loss(loss_dir)
        self.semantic_seg_criterion = build_loss(loss_seg)
        self.loss_ins = build_loss(loss_ins)
        self.loss_tda = build_loss(loss_tda)
        self.loss_geo = build_loss(loss_geo)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        #self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self._init_layers()
        self.use_tda = False
        self.loss_fn = torch.nn.BCELoss(reduction='mean')
        if self.query_embed_type == 'none_instance_pts':
            self.reid_embed_head = MLP(256, 256, 256, 3)
            self.memory_bank = MemoryBank(16)
            self.pre_memory_bank = MemoryBank_temp(16)
            self.pre_seg_map = None
            self.loss_seg_pre = torch.nn.L1Loss()
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)
            elif self.query_embed_type == 'none_instance_pts':
                self.query_embedding = None
                # self.instance_embedding = nn.Embedding(self.num_vec_one2many, self.embed_dims * 2)
                # self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)

    # @auto_fp16(apply_to=('mlvl_feats'))

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None,  only_bev=False,pre_feat=None,mask=None,train=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        if train:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        elif self.query_embed_type == 'none_instance_pts':
            #object_query_embeds = None
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)


        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None
            # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
                torch.zeros([num_vec, num_vec, ]).bool().to(mlvl_feats[0].device)
            )
        self_attn_mask[self.num_vec_one2one:, 0: self.num_vec_one2one, ] = True
        self_attn_mask[0: self.num_vec_one2one, self.num_vec_one2one:, ] = True

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
                pre_feat=pre_feat,
                mask_gt=mask,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
                #object_query_embeds_one2many=object_query_embeds_one2many,
        )

        bev_embed, hs, init_reference, inter_references,semantic_mask,instance_mask,depth = outputs#
        hs = hs.permute(0, 2, 1, 3)#6 4 50*20 256
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        outputs_pts_coords_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference[..., 0:2] if not self.z_cfg['gt_z_flag'] else init_reference[..., 0:3]
            else:
                reference = inter_references[lvl - 1][..., 0:2] if not self.z_cfg['gt_z_flag'] else inter_references[
                                                                                                        lvl - 1][...,
                                                                                          0:3]


            reference = inverse_sigmoid(reference)
            # import pdb;pdb.set_trace()
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            outputs_class = self.cls_branches[lvl](hs[lvl]
                                            .view(bs,num_vec, self.num_pts_per_vec,-1)
                                            .mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp+= reference
            tmp = tmp.sigmoid() # cx,cy,w,h
            # TODO: check if using sigmoid
            outputs_coord, outputs_pts_coord = self.transform_box_v2(tmp,num_vec=num_vec)
            outputs_classes.append(outputs_class[:, 0:self.num_vec_one2one])
            outputs_coords.append(outputs_coord[:, 0:self.num_vec_one2one])
            outputs_pts_coords.append(outputs_pts_coord[:, 0:self.num_vec_one2one])
            outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
            outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
            outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])
        hs = hs[-1].view(bs,num_vec, self.num_pts_per_vec,-1).mean(2)
        hs = self.reid_embed_head(hs)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
        outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
        outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])
        outs = {
            'depth':depth,
            'bev_embed': bev_embed,#4 200*100 256
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'hs':hs,#4 50 256
            'all_pts_preds': outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'enc_pts_preds': None,
            'semantic_mask':semantic_mask,
            'instance_mask': instance_mask,
            "one2many_outs": dict(
                all_cls_scores=outputs_classes_one2many,
                all_bbox_preds=outputs_coords_one2many,
                all_pts_preds=outputs_pts_coords_one2many,
                enc_cls_scores=None,
                enc_bbox_preds=None,
                enc_pts_preds=None,
                semantic_mask=None,
                instance_mask=None,
                pv_seg=None,
            )
        }

        return outs

    def transform_box_v2(self, pts, num_vec=50, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        if self.z_cfg['gt_z_flag']:
            pts_reshape = pts.view(pts.shape[0], num_vec,
                                   self.num_pts_per_vec, 3)
        else:
            pts_reshape = pts.view(pts.shape[0], num_vec,
                                   self.num_pts_per_vec, 2)

        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape
    def _get_target_single(self,
                           cls_score,#50 3
                           bbox_pred,#50 4
                           pts_pred,#50 20 2
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index,all_cost = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)#50 6

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds#预测正确的的序列
        neg_inds = sampling_result.neg_inds#预测错误的的序列

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)#50 ，全为3
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]#预测正确的改为gt的label，错误为3 sampling_result.pos_assigned_gt_inds 为对应的gt序号
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)#50 4
        bbox_weights[pos_inds] = 1.0#预测正确的bbox为1，错误为0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]#预测正确的bbox，所对应的gt
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes#预测正确的改为gtbbox，其他为0
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        gt_num = sampling_result.pos_assigned_gt_inds
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds,all_cost,gt_num)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list,all_cost,gt_num) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg,all_cost,gt_num,pos_inds_list)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None,
                    ):#所有batch一起
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg,num_all_cost,num_gt_num,pos_inds_list) = cls_reg_targets



        # import pdb;pdb.set_trace()
        labels = torch.cat(labels_list, 0)#B*50
        label_weights = torch.cat(label_weights_list, 0)#B*50
        bbox_targets = torch.cat(bbox_targets_list, 0)#B*50 4
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)#B*50 2
        pts_weights = torch.cat(pts_weights_list, 0)


        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # import pdb;pdb.set_trace()
        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()

        # num_samples, num_order, num_pts, num_coords
        #normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)#归一化
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else normalize_3d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        loss_pts = self.loss_pts(
            pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                            :,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        #denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
            else denormalize_3d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir, labels_list,pos_inds_list,num_all_cost,num_gt_num
    def seg_loss(self, predictions, targets):
        loss = self.semantic_seg_criterion(
            predictions.float(),
            targets.long().cuda(),
        )
        return loss


    def trans_into_now(self,prev_bev,img_metas):
        b = prev_bev.shape[1]
        prev_bev = rearrange(prev_bev, ' (h w) b c -> b c h w', h=200, w=100)

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
        p = torch.stack(map).cuda()  # b 256 200 100
        return p

    def pre_memory_update(self,gt_bboxes_list,
             gt_labels_list,
             preds_dicts,pre=False):
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        cls_scores = preds_dicts['all_cls_scores'][-1]
        bbox_preds = preds_dicts['all_bbox_preds'][-1]
        pts_preds = preds_dicts['all_pts_preds'][-1]
        embed = preds_dicts['hs']
        device = gt_labels_list[0].device
        gt_bboxes_ignore_list = None

        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        # import pdb;pdb.set_trace()
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg, num_all_cost, num_gt_num, pos_inds_list) = cls_reg_targets
        b = len(num_all_cost)
        label_0 = [dict() for i in range(b)]
        label_1 = [dict() for i in range(b)]
        label_2 = [dict() for i in range(b)]
        vec = dict()
        for i in range(b):
            vec[i] = {0: [], 1: [], 2: []}
        for i in range(len(num_all_cost)):
            for j in range(len(pos_inds_list[i])):  # 预测正确的序号
                row = pos_inds_list[i][j]
                gt_row = num_gt_num[i][j]  # 对应gt的序号
                label = labels_list[i][row]
                cost = int(num_all_cost[i][row][gt_row])
                point = bbox_preds_list[i][row]
                value = {'bbox': point, 'row': row}
                pts = pts_preds_list[i][row]
                pts = denormalize_2d_pts(pts, self.pc_range)
                cls_score = cls_scores_list[i][row]
                if label == 0:
                    label_0[i][cost] = value
                    if cls_score[0] > 0.4:
                        vec[i][0].append(LineString(pts))
                if label == 1:
                    label_1[i][cost] = value
                    if cls_score[1] > 0.4:
                        vec[i][1].append(LineString(pts))
                if label == 2:
                    label_2[i][cost] = value
                    if cls_score[2] > 0.4:
                        vec[i][2].append(LineString(pts))
        all_map = []
        for j in range(len(label_0)):
            batch_label_0 = sorted(label_0[j].items())
            for i in range(len(batch_label_0)):
                if i == 0:
                    self.pre_memory_bank.clean(0, j)
                if i < 5:
                    row = batch_label_0[i][1]['row']
                    v = embed[j, row].detach()
                    self.pre_memory_bank.update(0, v, j, batch_label_0[i][1]['bbox'])
                else:
                    break
            batch_label_1 = sorted(label_1[j].items())
            for i in range(len(batch_label_1)):
                if i == 0:
                    self.pre_memory_bank.clean(1, j)
                if i < 5:
                    row = batch_label_1[i][1]['row']
                    v = embed[j, row].detach()
                    self.pre_memory_bank.update(1, v, j, batch_label_1[i][1]['bbox'])
                else:
                    break
            batch_label_2 = sorted(label_2[j].items())
            for i in range(len(batch_label_2)):
                if i == 0:
                    self.pre_memory_bank.clean(2, j)
                if i < 5:
                    row = batch_label_2[i][1]['row']
                    v = embed[j, row].detach()
                    self.pre_memory_bank.update(2, v, j, batch_label_2[i][1]['bbox'])
                else:
                    break
            if pre:
                w = self.pc_range[3] - self.pc_range[0]
                h = self.pc_range[4] - self.pc_range[1]
                single_map = []
                for i in range(3):
                    if len(vec[j][i]) != 0:
                        map_mask = mask_line(vec[j][i], local_box=(0, 0, h, w),
                                             canvas_size=(self.bev_h, self.bev_w))  # 200 100
                    else:
                        map_mask = np.zeros((self.bev_h, self.bev_w))
                    single_map.append(map_mask)
                single_map = torch.tensor(single_map)
                t = single_map != 0
                single_map = torch.cat([((~torch.any(t, axis=0)).unsqueeze(0)) * 1, single_map])  # 4 200 100
                all_map.append(single_map)
        if pre:
            self.pre_seg_map = torch.stack(all_map)
            #self.show(self.pre_seg_map)

    def cmp_loss(self,loss_constract):
        return torch.log(torch.ones_like(loss_constract) + loss_constract)
    def get_v_and_neg_loss_single(self,label,embed,bbox_list,min_label,pos_lab_1,pos_lab_2,score):
        if len(label) == 0:
            return torch.zeros(1).cuda()
        l = min(len(label), 6)
        loss_constract = torch.zeros(1).cuda()
        n = 0
        for j in range(l):
            b = label[j][1]['b']
            row = label[j][1]['row']
            sf = score[b][row]
            # if sf[min_label] < 0.2:
            #     continue
            v = embed[b, row]
            if self.pre_memory_bank.is_notempty(min_label, b):
                v_bbox = bbox_list[b, row].unsqueeze(0)  #
                target_bbox = self.pre_memory_bank.get(min_label, b)
                # print(self.pre_memory_bank.is_notempty(min_label, b))
                target_bbox = torch.stack([target_bbox[t]['bbox'] for t in range(len(target_bbox))])  # x 4
                # num=target_bbox.shape[0]
                distance = target_bbox[:, 0:2] - v_bbox[:, 0:2]  # x 2
                dis = distance[:, 0] ** 2 + distance[:, 1] ** 2  # 3
                # _, n = torch.min(dis, dim=0)
                num = min(len(target_bbox), 2)
                values, indices = torch.topk(-dis, k=num)
                pos_embed_group = []
                for i in range(len(values)):
                    n = indices[i]
                    pos_emb = self.pre_memory_bank.get(min_label, b)[n]['emb']
                    pos_embed_group.append(pos_emb)
                loss_all = torch.zeros(1).cuda()
                p1 = torch.stack(self.memory_bank.get(pos_lab_1, False)).cuda()
                p2 = torch.stack(self.memory_bank.get(pos_lab_2, False)).cuda()
                neg_embed = torch.cat((p1, p2), dim=0).cuda()

                for i in range(len(pos_embed_group)):
                    pos_emb = pos_embed_group[i]
                    pos_mask = torch.ones_like(neg_embed) * pos_emb.unsqueeze(0)
                    v_mask = torch.ones_like(neg_embed) * v.unsqueeze(0)

                    #loss_c = v_mask @ neg_embed.T - v_mask @ pos_mask.T
                    loss_c = v_mask * neg_embed - v_mask *pos_mask
                    loss_c = torch.exp(loss_c)
                    loss_c = torch.mean(loss_c)
                    loss_all += loss_c

                loss_constract +=loss_all# self.cmp_loss(loss_all)  #
                n = n + 1
            else:
                pass

        if n > 0:
            loss_constract = loss_constract / n

        return loss_constract



    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             semantic_mask_gt, instance_mask_gt,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             prev_bev,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        # import pdb;pdb.set_trace()
        all_cls_scores = preds_dicts['all_cls_scores']
        scores = all_cls_scores[-1]
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds  = preds_dicts['enc_pts_preds']
        embed = preds_dicts['hs']
        if 'semantic_mask' in preds_dicts:
            semantic_mask = preds_dicts['semantic_mask']
            instance_mask = preds_dicts['instance_mask']
        else:
            semantic_mask = None
            instance_mask = None

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device


        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]
        # import pdb;pdb.set_trace()
        # gt_bboxes_list = [
        #     gt_bboxes.to(device) for gt_bboxes in gt_bboxes_list]
        gt_bboxes_list = [
            gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
        gt_pts_list = [
            gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        if self.gt_shift_pts_pattern == 'v0':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v1':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v2':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v3':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
        elif self.gt_shift_pts_pattern == 'v4':
            gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
        else:
            raise NotImplementedError
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
        all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # import pdb;pdb.set_trace()
        losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir,labels_list,num_total_pos,num_all_cost,num_gt_num = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,
            all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
            all_gt_bboxes_ignore_list)
        loss_dict = dict()

        use_constract = True
        if use_constract and self.query_embed_type=='none_instance_pts':
            bbox_list = all_bbox_preds[-1]
             # 4 50 256
            labels_list, num_total_pos, num_all_cost, num_gt_num,all_pts_preds = labels_list[-1],num_total_pos[-1],num_all_cost[-1],num_gt_num[-1],all_pts_preds[-1]
            # find match pair
            label_0 = dict()
            label_1 = dict()
            label_2 = dict()
            b = len(num_all_cost)
            vec = dict()
            for i in range(b):
                vec[i] = {0: [], 1: [], 2: []}
            for i in range(len(num_all_cost)):
                for j in range(len(num_total_pos[i])):
                    row = num_total_pos[i][j]
                    gt_row = num_gt_num[i][j]
                    label = labels_list[i][row]
                    cost = int(num_all_cost[i][row][gt_row])
                    value = {'b': i, 'row': row}
                    pts = all_pts_preds[i][row]
                    pts = denormalize_2d_pts(pts,self.pc_range)
                    cls_score = scores[i][row]
                    if label == 0:
                        label_0[cost] = value
                        if cls_score[0] > 0.3:
                            vec[i][0].append(LineString(pts))
                    if label == 1:
                        label_1[cost] = value
                        if cls_score[1] > 0.3:
                            vec[i][1].append(LineString(pts))
                    if label == 2:
                        label_2[cost] = value
                        if cls_score[2] > 0.3:
                            vec[i][2].append(LineString(pts))
            label_0 = sorted(label_0.items())
            label_1 = sorted(label_1.items())
            label_2 = sorted(label_2.items())
            #first update
            if self.memory_bank.is_notempty(0) or self.memory_bank.is_notempty(1) or self.memory_bank.is_notempty(2):
                for i in range(len(label_0)):
                    if i < 5:
                        b = label_0[i][1]['b']
                        row = label_0[i][1]['row']
                        v = embed[b, row].detach()
                        self.memory_bank.update(0, v)
                    else:
                        break
                for i in range(len(label_1)):
                    if i < 5:
                        b = label_1[i][1]['b']
                        row = label_1[i][1]['row']
                        v = embed[b, row].detach()
                        self.memory_bank.update(1, v)
                    else:
                        break
                for i in range(len(label_2)):
                    if i < 5:
                        b = label_2[i][1]['b']
                        row = label_2[i][1]['row']
                        v = embed[b, row].detach()
                        self.memory_bank.update(2, v)
                    else:
                        break
            else:
                pass
            loss_0 = self.get_v_and_neg_loss_single(label_0,embed,bbox_list,0,1,2,scores )
            loss_1 = self.get_v_and_neg_loss_single(label_1,embed,bbox_list,1,0,2,scores)
            loss_2 = self.get_v_and_neg_loss_single(label_2,embed,bbox_list,2,0,1,scores)

            loss_constract=loss_2+loss_1+loss_0
            loss_constract = torch.log(torch.ones_like(loss_constract) + loss_constract)
            loss_dict['loss_com'] = 0.1*loss_constract
             ###update
            for i in range(len(label_0)):
                 if i == 0:
                     self.memory_bank.clean(0)
                 if i < 5:
                     b = label_0[i][1]['b']
                     row = label_0[i][1]['row']
                     v = embed[b, row].detach()
                     self.memory_bank.update(0, v)
                 else:
                     break
            for i in range(len(label_1)):
                 if i == 0:
                     self.memory_bank.clean(1)
                 if i < 5:
                     b = label_1[i][1]['b']
                     row = label_1[i][1]['row']
                     v = embed[b, row].detach()
                     self.memory_bank.update(1, v)
                 else:
                     break
            for i in range(len(label_2)):
                 if i == 0:
                     self.memory_bank.clean(2)
                 if i < 5:
                     b = label_2[i][1]['b']
                     row = label_2[i][1]['row']
                     v = embed[b, row].detach()
                     self.memory_bank.update(2, v)
                 else:
                     break

            ##now seg map update
            w = self.pc_range[3] - self.pc_range[0]
            h = self.pc_range[4] - self.pc_range[1]
            if True:
                all_map=[]
                for i in range(len(vec)):
                    single_map = []
                    for j in range(3):
                        if len(vec[i][j]) != 0:
                            map_mask =  mask_line(vec[i][j],local_box=(0,0,h,w),canvas_size=(self.bev_h, self.bev_w))  # 200 100
                        else:
                            map_mask = np.zeros((self.bev_h, self.bev_w))
                        single_map.append(map_mask)
                    single_map = torch.tensor(single_map)
                    t = single_map != 0
                    single_map = torch.cat([((~torch.any(t, axis=0)).unsqueeze(0)) * 1, single_map])  # 4 200 100
                    all_map.append(single_map)
                all_map = torch.stack(all_map).cuda()
                seg_loss = self.loss_pre_map(all_map,self.pre_seg_map,img_metas)
                loss_dict['loss_pre'] = 0.1*seg_loss


        if semantic_mask is not None :
            semantic_mask_gt = torch.stack(semantic_mask_gt)
            loss_seg = self.seg_loss(semantic_mask, semantic_mask_gt)
            loss_dict['loss_seg'] = 0.1*loss_seg

            var_loss, dist_loss, reg_loss = self.loss_ins(instance_mask.float(), instance_mask_gt)
            loss_inss = 1*(var_loss+dist_loss)
            loss_dict['loss_ins'] = 1*loss_inss

        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            # TODO bug here
            enc_loss_cls, enc_losses_bbox, enc_losses_iou, enc_losses_pts, enc_losses_dir = \
                self.loss_single(enc_cls_scores, enc_bbox_preds, enc_pts_preds,
                                 gt_bboxes_list, binary_labels_list, gt_pts_list,gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_losses_iou'] = enc_losses_iou
            loss_dict['enc_losses_pts'] = enc_losses_pts
            loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        loss_dict['loss_pts'] = losses_pts[-1]
        loss_dict['loss_dir'] = losses_dir[-1]
        #loss_dict['loss_geo_2'] = loss_geo_2[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        w=0.1
        for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1],
                                            ):
            if num_dec_layer == 0:
                loss_dict[f'd{num_dec_layer}.loss_cls'] = w*loss_cls_i
                loss_dict[f'd{num_dec_layer}.loss_bbox'] = w*loss_bbox_i
                loss_dict[f'd{num_dec_layer}.loss_iou'] = w*loss_iou_i
                loss_dict[f'd{num_dec_layer}.loss_pts'] = w*loss_pts_i
            else:
                loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.loss_bbox'] =loss_bbox_i
                loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
                loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i
                loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
            num_dec_layer += 1
        return loss_dict
    def get_mask(self,outs):
        bbox_list = self.get_bboxes(outs)  # 4list
        res_scores = []
        res_labels = []
        res_pts = []
        for bboxes, scores, labels, pts in bbox_list:
            res_scores.append(scores.cpu())
            res_labels.append(labels.cpu())
            res_pts.append(pts.to('cpu'))

        res_pts = torch.stack(res_pts)  # 4 50 20 2
        res_labels = torch.stack(res_labels)  # 4 50
        res_scores = torch.stack(res_scores)  # 4 50
        keep = res_scores > 0.0  # 4
        B, _ = res_labels.shape
        vec = dict()
        for i in range(B):
            vec[i] = dict()
            for j in range(3):
                vec[i][j] = []
        res_labels[res_scores<0.4] = -1
        cls_1 = torch.where(res_labels[:] == 0)  # x
        if len(cls_1) != 0:
            vec = trans(cls_1[0], cls_1[1], res_pts, vec, name=0)
        cls_2 = torch.where(res_labels[:] == 1)  # x y
        if len(cls_2) != 0:
            vec = trans(cls_2[0], cls_2[1], res_pts, vec, name=1)
        cls_3 = torch.where(res_labels[:] == 2)
        if len(cls_3) != 0:
            vec = trans(cls_3[0], cls_3[1], res_pts, vec, name=2)
        all_map = []
        w = self.pc_range[3]-self.pc_range[0]
        h = self.pc_range[4]-self.pc_range[1]
        for i in range(B):
            single_map = []
            for j in range(3):
                if len(vec[i][j]) != 0:
                    map_mask = mask_line(vec[i][j],local_box=(0,0,h,w),canvas_size=(self.bev_h, self.bev_w))  # 200 100
                else:
                    map_mask = np.zeros((self.bev_h, self.bev_w))
                single_map.append(map_mask)
            single_map = torch.tensor(single_map)
            t = single_map != 0
            single_map = torch.cat([((~torch.any(t, axis=0)).unsqueeze(0))*1, single_map])  # 4 200 100
            all_map.append(single_map)
        all_map = torch.stack(all_map)  # b  4 200 100
        all_map = all_map.cuda()
        return all_map

    def loss_vec_sec(self,outs,semantic_gt,loss):
        all_map = self.get_mask(outs)#4 4 200 100
        #self.show(all_map)
        semantic_gt = torch.stack(semantic_gt)
        # self.show(semantic_gt)
        # semantic_gt = (1*semantic_gt).double().long()
        # loss_seg = self.loss_fn(all_map,semantic_gt)
        loss_seg = self.seg_loss(all_map,semantic_gt)
        loss['loss_vec_seg'] = 0.01*loss_seg
        return loss

    def show(self,img):
        c = img.shape[1]
        img = img.cpu().numpy()#4 4 400 200
        plt.figure(figsize=(2, 4), dpi=100)
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        if c>3:
            plt.imshow(img[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(img[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(img[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        else:
            plt.imshow(img[0][0], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(img[0][1], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(img[0][2], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        plt.show()

    def loss_pre_map(self,now_map,pre_map,img_metas):

        b, c, _, _ = pre_map.shape

        now_rot = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in img_metas])#角度当前时刻的位姿
        delta_x = np.array([each['can_bus'][0] for each in img_metas])  # [bs] 前后两帧绝对距离的x分量-世界坐标系  正向为正值，反向为负值
        delta_y = np.array([each['can_bus'][1] for each in img_metas])  # [bs] 前后两帧绝对距离的y分量-世界坐标系  正向为正值，反向为负值
        delta_translation_global = torch.tensor(
            [delta_x, delta_y]).permute(1, 0).reshape(b, 2, 1)#b 2 1

        curr_transform_matrix = torch.tensor([[np.cos(-now_rot), -np.sin(-now_rot)],
                                              [np.sin(-now_rot), np.cos(-now_rot)]])  # (2, 2，bs) 世界坐标转换为当前帧自车坐标
        curr_transform_matrix = curr_transform_matrix.permute(2,0,1)#b 2 2
        delta_translation_curr = torch.matmul(
            curr_transform_matrix,
            delta_translation_global).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
        delta_translation_curr = delta_translation_curr // 0.3  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
        delta_translation_curr = delta_translation_curr.round().tolist()

        rot = np.array([each['can_bus'][-1] for each in img_metas])#两帧相差角度，4

        map =[]
        for i in range(b):
            tran = [-delta_translation_curr[i][0], delta_translation_curr[i][1]]
            pre_map_temp= pre_map[i]
            pre_warp = affine(pre_map_temp,  # 必须是(c, h, w)形式的tensor
                              angle=-rot[i],  # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                              translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                              scale=1,  # 浮点数，中心缩放尺度
                              shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                              interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                              fill=[0.0])
            pre_warp = torch.tensor(pre_warp)
            map.append(pre_warp)

        pre_map_warp = torch.stack(map).cuda()
        #l1 loss
        loss_seg = self.loss_seg_pre(pre_map_warp[:,1:].float(),now_map[:,1:].float())
        show = False
        if show:
            pre_map_warp = pre_map_warp.detach().cpu().numpy()
            now_map = now_map.detach().cpu().numpy()
            map = pre_map_warp+now_map
            plt.figure()
            plt.axis('off')  # 去坐标轴
            plt.xticks([])  # 去 x 轴刻度
            plt.yticks([])  # 去 y 轴刻度
            plt.subplot(131)
            plt.imshow(pre_map_warp[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(pre_map_warp[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(pre_map_warp[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
            plt.subplot(132)
            plt.imshow(now_map[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(now_map[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(now_map[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
            plt.subplot(133)
            plt.imshow(map[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
            plt.imshow(map[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
            plt.imshow(map[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
            plt.show()

        # pre_mask = torch.sum(pre_map_warp[:,1:],dim=1)
        # now_mask = torch.sum(now_map[:, 1:], dim=1)
        # pre_mask = pre_mask!=0#b 200 100
        # now_mask = now_mask!=0
        # interest = pre_mask*now_mask
        #interest = torch.sum(interest,dim=-1)

        #iou 损失
        # union = torch.sum(pre_mask,dim=-1)+torch.sum(now_mask,dim=-1)-interest
        # iou = interest/(union+torch.ones_like(union)*1e-8)
        # loss_seg = torch.ones_like(iou)-iou
        #dice 损失
        # union = torch.sum(pre_mask, dim=-1) + torch.sum(now_mask, dim=-1)
        # dice = (2 * interest + torch.ones_like(interest)) / (union + torch.ones_like(interest))#4 3
        # loss_seg = torch.ones_like(dice)-dice


        return loss_seg

    def get_lable(self,out):
        bbox_list = self.get_bboxes(out)  # 4list
        res_scores = []
        res_labels = []
        res_pts = []
        res_bbbox=[]
        for bboxes, scores, labels, pts in bbox_list:
            res_scores.append(scores.cpu())
            res_labels.append(labels.cpu())
            res_pts.append(pts.to('cpu'))
            res_bbbox.append(res_bbbox.cpu())
        res_pts = torch.stack(res_pts)  # 4 50 20 2
        res_labels = torch.stack(res_labels)  # 4 50
        res_scores = torch.stack(res_scores)  # 4 50
        res_bbbox = torch.stack(res_bbbox)
        return res_labels,res_pts,res_scores,res_bbbox

    def get_indice(self,now_res_labels,pre_res_labels,id,now_res_bbbox,pre_res_bbbox,now_embeds,pre_embeds):

        pos_0_now = float('inf')*(now_res_labels == id)  # 4 50
        pos_0_pre = float('-inf')*(pre_res_labels == id)

        now_res_bbbox = pos_0_now+now_res_bbbox
        pre_res_bbbox = pos_0_pre+pre_res_bbbox
        dist = torch.cdist(now_res_bbbox[:, 0:2],pre_res_bbbox[:, 0:2])  # 4 74369 74325
        dist1, n = torch.min(dist, 2)  # 4 74369
        _,now_n = torch.min(dist1,1)
        v = now_embeds[:,now_n]
        pos_n = n[now_n]
        pos_emb = pre_embeds[:,pos_n]


    def loss_constract(self,outs,pre,losses):
        now_embeds = self.reid_embed_head(outs['hs'])#4 1000 264
        pre_embeds = self.reid_embed_head(pre['hs'])#4 1000 264
        now_res_labels,now_res_pts,now_res_scores,now_res_bbbox = self.get_lable(outs)#4 50
        pre_res_labels,pre_res_pts,pre_res_scores,pre_res_bbbox = self.get_lable(pre)#4 50
        b,q,p,_ = now_res_pts.shape
        c = outs['hs'].shape[2]
        now_res_labels[now_res_scores < 0.4] = -1#4 50
        pre_res_labels[pre_res_scores < 0.4] = -1

        pos_0 = self.get_indice(now_res_labels,pre_res_labels,0,now_res_bbbox,pre_res_bbbox,now_embeds,pre_embeds)
        pos_1 = self.get_indice(now_res_labels,pre_res_labels,1,now_res_bbbox,pre_res_bbbox,now_embeds,pre_embeds)
        pos_2 = self.get_indice(now_res_labels, pre_res_labels, 2,now_res_bbbox,pre_res_bbbox,now_embeds,pre_embeds)
        #neg_0_3 = self.get_indice(now_res_labels, pre_res_labels, -1, pred)
        loss_1 = torch.exp(pos_1-pos_0)+torch.exp(pos_2-pos_0)#+torch.exp(neg_0_3-pos_0)

        loss_2 = torch.exp(pos_0-pos_1)+torch.exp(pos_2-pos_1)#+torch.exp(neg_0_3-pos_0)

        loss_3 = torch.exp(pos_1-pos_2)+torch.exp(pos_0-pos_2)#+torch.exp(neg_0_3-pos_0)

        loss = loss_3+loss_2+loss_1
        loss = torch.log(1+loss)

        losses['loss_pre_map'] = 0.5*loss
        return losses


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas=None, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0#-15
    y_min = patch_y - patch_h / 2.0#-15
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(
        patch_x, patch_y), use_radians=False)

    return patch
def mask_for_lines(new_line,map_mask,idx):
    coords = np.asarray(list(new_line.coords), np.int32)
    coords = coords.reshape((-1, 2))#50 2
    #coords = np.sort(coords, axis=0)
    cv2.polylines(map_mask, [coords], False, color=idx, thickness=5)
    return map_mask,idx
def mask_line(layer_geom, local_box,canvas_size,idx=1):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)  # (-30,-15,30,15)

    canvas_h = canvas_size[0]  # 400
    canvas_w = canvas_size[1]  # 200
    scale_height = canvas_h / patch_h  # 0.15
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros((canvas_h, canvas_w), np.uint8)
    for line in layer_geom:
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
        if new_line.geom_type == 'MultiLineString':
            for new_single_line in new_line:
                map_mask, idx = mask_for_lines(
                    new_single_line, map_mask, idx)
        else:
            map_mask, idx = mask_for_lines(
                new_line, map_mask, idx)

    return map_mask
def trans(h,w,target,vec,name):
    h =h.cpu().numpy()
    w = w.cpu().numpy()
    target = target.detach().cpu().numpy()
    for i in range(h.shape[0]):#b
        t0 = target[h[i]][w[i]]
        if name!=1:
            vec[h[i]][name].append(LineString(t0))
        else:
            vec[h[i]][name].append(LinearRing(t0))

    return vec
if __name__ == '__main__':
    vec=dict()
    for i in range(4):
        vec[i] = dict()
        for j in range(3):
            vec[i][j] = []
    res_labels = torch.randn((4,50))
    res_pts= torch.randn((4,50,20,2))
    cls_1 = torch.where(res_labels[:] > 0)#2 hang lie
    vec = trans(cls_1[0], cls_1[1], res_pts, vec, name=0)