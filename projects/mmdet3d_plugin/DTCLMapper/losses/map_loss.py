# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
import mmcv
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import functools
import numpy
from shapely.geometry import LineString, box
from ..topo_data.tda.persistent_homology import buildGraph,ripsFiltration,filterBoundaryMatrix,reduceBoundaryMatrix,readPersistence,readIntervals
def graph_ph(persistence, homology_group = 0):
    # this function just produces the barcode graph for each homology group
    xstart = [s[1][0] for s in persistence if s[0] == homology_group]
    xstop = [s[1][1] for s in persistence if s[0] == homology_group]
    x = [xstop[i]-xstart[i] for i in range(len(xstart))]
    return x

def tda(data):
    graph = buildGraph(raw_data=data, epsilon=10)
    ripsComplex2 = ripsFiltration(graph, k=2)
    bm = filterBoundaryMatrix(ripsComplex2)
    z = reduceBoundaryMatrix(bm)  # 方法返回两个矩阵，减化边界矩阵和记录简化算法所有动作的记忆矩阵。
    intervals = readIntervals(z, ripsComplex2[1])
    persist1 = readPersistence(intervals, ripsComplex2)
    tda_0 = graph_ph(persist1,0)
    tda_1 = graph_ph(persist1, 1)
    return tda_0,tda_1
def tda_loss(a,b):
    if len(a)==0 and len(b)==0:
        return torch.tensor([0]).cuda()
    elif len(a)==0:
        b = torch.tensor(b).unsqueeze(1).to(torch.float32).cuda()
        loss,_ = torch.max(b,dim=0)
        return loss
    elif len(b)==0:
        a = torch.tensor(a).unsqueeze(1).to(torch.float32).cuda()
        loss, _ = torch.max(a, dim=0)
        return loss
    else:
        a = torch.tensor(a).unsqueeze(1).to(torch.float32).cuda()
        b = torch.tensor(b).unsqueeze(1).to(torch.float32).cuda()
        dist_1 = torch.cdist(a,b)#20 20
        dist_1,_ = torch.min(dist_1,dim=1)
        loss_1,_ = torch.max(dist_1,dim=0)
        loss_1 = torch.abs(loss_1)

        dist_2 = torch.cdist(b, a)
        dist_2,_ = torch.min(dist_2, dim=1)
        loss_2,_ = torch.max(dist_2, dim=0)
        loss_2 = torch.abs(loss_2)

        if loss_1>loss_2:
            return loss_1
        else:
            return loss_2
def creat_less_point(coords):
    coords = numpy.sort(coords, axis=1)
    vec = LineString(coords)
    distances = numpy.linspace(0, vec.length, 4)
    # if vec.length <5.5:
    #     distances = numpy.linspace(0, vec.length, 3)
    # elif vec.length <10.5:
    #     distances = numpy.linspace(0, vec.length, 5)
    # else:
    #     distances = numpy.linspace(0, vec.length, 10)
    sampled_points = numpy.array([list(vec.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    return sampled_points
@LOSSES.register_module()
class TDALoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(TDALoss, self).__init__()
        self.loss_weight = loss_weight
    def forward(self,pred,target,device):
        N,num_pts,num_coords = pred.shape
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        loss_0 = torch.zeros((1)).cuda(device)
        loss_1 = torch.zeros((1)).cuda(device)
        for i in range(N):
            #print(i)
            data1 = pred[i]#20 2
            data2 = target[i]
            mean = numpy.mean(data2)
            if mean == 0.5:
                continue
            else:
                data2 = creat_less_point(data2)
                data1 = creat_less_point(data1)

            pred_tda_0,pred_tda_1 = tda(data1)
            target_tda_0, target_tda_1 = tda(data2)
            tda_loss_0 = tda_loss(pred_tda_0,target_tda_0)
            tda_loss_1 = tda_loss(pred_tda_1, target_tda_1)
            loss_0 += tda_loss_0
            loss_1 += tda_loss_1
        return loss_0,loss_1
@LOSSES.register_module()
class DiscriminativeLoss(nn.Module):
    def __init__(self, embed_dim, delta_v, delta_d):
        super(DiscriminativeLoss, self).__init__()
        self.embed_dim = embed_dim
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(self, embedding, seg_gt):
        if embedding is None:
            return 0, 0, 0
        bs = embedding.shape[0]

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(bs):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            # labels = torch.unique(seg_gt_b)
            # labels = labels[labels != 0]
            num_lanes = seg_gt_b.shape[0]
            #assert num_lanes<51
            if num_lanes == 0:
                # please refer to issue here: https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for i in range(num_lanes):
                seg_mask_i = seg_gt_b[i].bool()
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim, 1), dim=0) - self.delta_v) ** 2) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)
            num_lanes=centroid_mean.shape[0]
            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device) * self.delta_d  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / bs
        dist_loss = dist_loss / bs
        reg_loss = reg_loss / bs
        return var_loss, dist_loss, reg_loss
# @LOSSES.register_module()
# class SimpleLoss(torch.nn.Module):
#     def __init__(self, pos_weight):
#         super(SimpleLoss, self).__init__()
#         self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))#内部含有sigmod
#
#     def forward(self, ypred, ytgt):
#         loss = self.loss_fn(ypred, ytgt)
#         return loss
@LOSSES.register_module()
class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0,weight=1.0):

        super().__init__()
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights).float()
        else: self.class_weights =class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.weight = weight
        # self.ce_criterion = nn.CrossEntropyLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

        # self.nll_criterion = nn.NLLLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

    def forward(self, prediction, target):
        b,c, h, w = prediction.shape
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float() if self.class_weights is not None else None,
            ignore_index=self.ignore_index,
        )

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b,  -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return self.weight*loss.mean()

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_dir
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            # loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum()
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_order, num_pts, num_coords
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum((1,2,3))
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def custom_weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weighted_dir_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_smooth_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss =smooth_l1_loss(pred,target, reduction='none')
    # import pdb;pdb.set_trace()
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss

@LOSSES.register_module()
class OrderedPtsSmoothL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_dir



@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@LOSSES.register_module()
class OrderedPtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # import pdb;pdb.set_trace()
        loss_bbox = self.loss_weight * ordered_pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox




@MATCH_COST.register_module()
class OrderedPtsSmoothL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1).unsqueeze(1).repeat(1,num_gts*num_orders,1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1).unsqueeze(0).repeat(bbox_pred.size(0),1,1)
        # import pdb;pdb.set_trace()
        bbox_cost = smooth_l1_loss(bbox_pred, gt_bboxes, reduction='none').sum(-1)
        # bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class PtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.view(num_gts,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class MyChamferDistanceCost:
    def __init__(self, loss_src_weight=1., loss_dst_weight=1.):
        # assert mode in ['smooth_l1', 'l1', 'l2']
        # self.mode = mode
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def __call__(self, src, dst,src_weight=1.0,dst_weight=1.0,):
        """
        pred_pts (Tensor): normed coordinate(x,y), shape (num_q, num_pts_M, 2)
        gt_pts (Tensor): normed coordinate(x,y), shape (num_gt, num_pts_N, 2)
        """
        # criterion_mode = self.mode
        # if criterion_mode == 'smooth_l1':
        #     criterion = smooth_l1_loss
        # elif criterion_mode == 'l1':
        #     criterion = l1_loss
        # elif criterion_mode == 'l2':
        #     criterion = mse_loss
        # else:
        #     raise NotImplementedError
        # import pdb;pdb.set_trace()
        src_expand = src.unsqueeze(1).repeat(1,dst.shape[0],1,1)
        dst_expand = dst.unsqueeze(0).repeat(src.shape[0],1,1,1)
        # src_expand = src.unsqueeze(2).unsqueeze(1).repeat(1,dst.shape[0], 1, dst.shape[1], 1)
        # dst_expand = dst.unsqueeze(1).unsqueeze(0).repeat(src.shape[0],1, src.shape[1], 1, 1)
        distance = torch.cdist(src_expand, dst_expand)
        src2dst_distance = torch.min(distance, dim=3)[0]  # (num_q, num_gt, num_pts_N)
        dst2src_distance = torch.min(distance, dim=2)[0]  # (num_q, num_gt, num_pts_M)
        loss_src = (src2dst_distance * src_weight).mean(-1)
        loss_dst = (dst2src_distance * dst_weight).mean(-1)
        loss = loss_src*self.loss_src_weight + loss_dst * self.loss_dst_weight
        return loss

@mmcv.jit(derivate=True, coderize=True)
def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                    #  criterion_mode='l1',
                     reduction='mean',
                     avg_factor=None):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    # if criterion_mode == 'smooth_l1':
    #     criterion = smooth_l1_loss
    # elif criterion_mode == 'l1':
    #     criterion = l1_loss
    # elif criterion_mode == 'l2':
    #     criterion = mse_loss
    # else:
    #     raise NotImplementedError

    # src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    # dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # import pdb;pdb.set_trace()
    distance = torch.cdist(src, dst)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
    # import pdb;pdb.set_trace()
    #TODO this may be wrong for misaligned src_weight, now[N,fixed_num]
    # should be [N], then view
    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            raise ValueError('MyCDLoss can not be used with reduction=`none`')
        elif reduction_enum == 1:
            loss_src = loss_src.mean(-1).mean()
            loss_dst = loss_dst.mean(-1).mean()
        elif reduction_enum == 2:
            loss_src = loss_src.mean(-1).sum()
            loss_dst = loss_dst.mean(-1).sum()
        else:
            raise NotImplementedError
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss_src = loss_src.mean(-1).sum() / (avg_factor + eps)
            loss_dst = loss_dst.mean(-1).sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss_src, loss_dst, indices1, indices2


@LOSSES.register_module()
class MyChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                #  mode='l1',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0):
        super(MyChamferDistance, self).__init__()

        # assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        # self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(self,
                source,
                target,
                src_weight=1.0,
                dst_weight=1.0,
                avg_factor=None,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, reduction,
            avg_factor=avg_factor)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        loss_pts = loss_source + loss_target

        if return_indices:
            return loss_pts, indices1, indices2
        else:
            return loss_pts
