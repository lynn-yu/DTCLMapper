import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import argparse
import mmcv

import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
import sys
sys.path.append('')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
from shapely.geometry import LineString
import cv2
import copy
from einops import rearrange
from shapely.geometry import LineString, box,LinearRing
from shapely import affinity
import matplotlib.pyplot as plt
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
from scipy.spatial import Voronoi, voronoi_plot_2d
import open3d as o3d
caption_by_cam={
    'ring_front_center':'CAM_FRONT_CENTER',
    'ring_front_right':'CAM_FRONT_RIGHT',
    'ring_front_left': 'CAM_FRONT_LEFT',
    'ring_rear_right': 'CAM_REAR_RIGHT',
    'ring_rear_left': 'CAM_REAT_LEFT',
    'ring_side_right': 'CAM_SIDE_RIGHT',
    'ring_side_left': 'CAM_SIDE_LEFT',
}
COLOR_MAPS_BGR = {
    # bgr colors
    'divider': (54,137,255),
    'boundary': (0, 0, 255),
    'ped_crossing': (255, 0, 0),
    'centerline': (0,255,0),
    'drivable_area': (171, 255, 255)
}

data_path_prefix = '/home/users/yunchi.zhang/project/MapTR' # project root

def remove_nan_values(uv):
    is_u_valid = np.logical_not(np.isnan(uv[:, 0]))
    is_v_valid = np.logical_not(np.isnan(uv[:, 1]))
    is_uv_valid = np.logical_and(is_u_valid, is_v_valid)

    uv_valid = uv[is_uv_valid]
    return uv_valid

def interp_fixed_dist(line, sample_dist):
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

def draw_visible_polyline_cv2(line, valid_pts_bool, image, color, thickness_px,map_class):
    """Draw a polyline onto an image using given line segments.
    Args:
        line: Array of shape (K, 2) representing the coordinates of line.
        valid_pts_bool: Array of shape (K,) representing which polyline coordinates are valid for rendering.
            For example, if the coordinate is occluded, a user might specify that it is invalid.
            Line segments touching an invalid vertex will not be rendered.
        image: Array of shape (H, W, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        thickness_px: thickness (in pixels) to use when rendering the polyline.
    """
    line = np.round(line).astype(int)  # type: ignore
#     if map_class == 'centerline':
#         instance = LineString(line).simplify(0.2, preserve_topology=True)
#         line = np.array(list(instance.coords))
#         line = np.round(line).astype(int)
    for i in range(len(line) - 1):

        if (not valid_pts_bool[i]) or (not valid_pts_bool[i + 1]):
            continue

        x1 = line[i][0]
        y1 = line[i][1]
        x2 = line[i + 1][0]
        y2 = line[i + 1][1]

        # Use anti-aliasing (AA) for curves
        if map_class != 'centerline':
            image = cv2.line(image, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness_px, lineType=cv2.LINE_AA)
        else:
            image = cv2.arrowedLine(image,(x1, y1),(x2,y2),color,thickness_px,8,0,0.7)


def points_ego2img(pts_ego, lidar2img):
    pts_ego_4d = np.concatenate([pts_ego, np.ones([len(pts_ego), 1])], axis=-1)
    pts_img_4d = lidar2img @ pts_ego_4d.T
    
    
    uv = pts_img_4d.T
    uv = remove_nan_values(uv)
    depth = uv[:, 2]
    uv = uv[:, :2] / uv[:, 2].reshape(-1, 1)

    return uv, depth
def draw_polyline_ego_on_img(polyline_ego, img_bgr, lidar2img, map_class, thickness):
    # if 2-dimension, assume z=0
    if polyline_ego.shape[1] == 2:
        zeros = np.zeros((polyline_ego.shape[0], 1))
        polyline_ego = np.concatenate([polyline_ego, zeros], axis=1)

    polyline_ego = interp_fixed_dist(line=LineString(polyline_ego), sample_dist=0.2)
    
    uv, depth = points_ego2img(polyline_ego, lidar2img)

    h, w, c = img_bgr.shape

    is_valid_x = np.logical_and(0 <= uv[:, 0], uv[:, 0] < w - 1)
    is_valid_y = np.logical_and(0 <= uv[:, 1], uv[:, 1] < h - 1)
    is_valid_z = depth > 0
    is_valid_points = np.logical_and.reduce([is_valid_x, is_valid_y, is_valid_z])

    if is_valid_points.sum() == 0:
        return
    
    tmp_list = []
    for i, valid in enumerate(is_valid_points):
        
        if valid:
            tmp_list.append(uv[i])
        else:
            if len(tmp_list) >= 2:
                tmp_vector = np.stack(tmp_list)
                tmp_vector = np.round(tmp_vector).astype(np.int32)
                draw_visible_polyline_cv2(
                    copy.deepcopy(tmp_vector),
                    valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
                    image=img_bgr,
                    color=COLOR_MAPS_BGR[map_class],
                    thickness_px=thickness,
                    map_class=map_class
                )
            tmp_list = []
    if len(tmp_list) >= 2:
        tmp_vector = np.stack(tmp_list)
        tmp_vector = np.round(tmp_vector).astype(np.int32)
        draw_visible_polyline_cv2(
            copy.deepcopy(tmp_vector),
            valid_pts_bool=np.ones((len(uv), 1), dtype=bool),
            image=img_bgr,
            color=COLOR_MAPS_BGR[map_class],
            thickness_px=thickness,
            map_class=map_class,
        )

def render_anno_on_pv(cam_img, anno, lidar2img):
    for key, value in anno.items():
        for pts in value:
            draw_polyline_ego_on_img(pts, cam_img, lidar2img, 
                       key, thickness=10)

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('--config',default="/data0/lsy/4.v2_MapTR-main/work_dirs/run_18/v4maptr_tiny_r50_24e.py", help='test config file path')
    parser.add_argument('--checkpoint',
                        #default='/data0/lsy/4.v2_MapTR-main/work_dirs/run_18/epoch_5.pth',#maptr
                        default ='/data0/lsy/4.v2_MapTR-main/work_dirs/run_21/epoch_6.pth',#our
                        #default = '/data0/lsy/4.v2_MapTR-main/work_dirs/run_23/epoch_6.pth',
                         help='checkpoint file'
                        )
    parser.add_argument('--score-thresh', default=0.3, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('/data0/lsy/4.v2_MapTR-main/figs/lidar_car.png')

    # get color map: divider->orange, ped->blue, boundary->red, centerline->green
    colors_plt = ['orange', 'blue', 'red','green']

    logger.info('BEGIN vis test dataset samples gt label & pred')

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    final_dict = {}
    f = False
    t = True
    show = t
    show2=f
    restore = f
    pre_semantic = None
    pre_pose=None
    for i, data in enumerate(data_loader):
        if i>11 or i<9:
            continue
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue

        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.split('.')[0]
        # import pdb;pdb.set_trace()
        # if pts_filename not in CANDIDATE:
        #     continue
        sample_dict = {}
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))
        semantic_gt = data['semantic_gt'].data[0]  # 4 200 100

        #reference = rearrange(new_prev_bev['all_pts_preds'],'n b i p c -> n b (i p) c')
        # for lvl in range(6):
        #     if lvl == 5: #or lvl == 1 or lvl == 2 or lvl == 3:
        #         points = reference[lvl]
        #         x = (points[0, :, 0] * 200).cpu().numpy()
        #         y = (points[0, :, 1] * 100).cpu().numpy()
        #         fig = plt.figure()  # 创建画布
        #         plt.axis('off')  # 去坐标轴
        #         plt.xticks([])  # 去 x 轴刻度
        #         plt.yticks([])  # 去 y 轴刻度
        #         plt.imshow(~semantic_gt[0][0], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        #         plt.plot(x, y, 'ro')
        #         semantic_path = osp.join(sample_dir, 'point_layer_'+str(lvl)+'.png')
        #         if show2:
        #             plt.show()
        #         else:
        #             plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)

        #semantic_gt = rearrange(semantic_gt, 'c h w  -> b h w c')
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(semantic_gt[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(semantic_gt[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(semantic_gt[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        semantic_path = osp.join(sample_dir, 'semantic_gt.png')
        if show:
            plt.show()
        else:
            plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)
        # instance = data['instance_mask'].data[0]
        # plt.figure(figsize=(1, 2), dpi=100)
        # plt.axis('off')  # 去坐标轴
        # plt.xticks([])  # 去 x 轴刻度
        # plt.yticks([])  # 去 y 轴刻度
        # plt.imshow(instance[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        # plt.imshow(instance[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        # plt.imshow(instance[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        # if show:
        #     plt.show()
        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        # import ipdb;ipdb.set_trace() 
        for filepath, lidar2img, img_aug in zip(filename_list,img_metas[0]['lidar2img'],img_metas[0]['img_aug_matrix']):
            inv_aug = np.linalg.inv(img_aug)
            lidar2orimg = np.dot(inv_aug, lidar2img)
            cam_name = os.path.dirname(filepath).split('/')[-1]
            img_path_dict[cam_name] = dict(
                filepath=filepath,
                lidar2img = lidar2orimg)
        sample_dict['imgs_path'] = img_path_dict
        gt_dict = {'divider':[],'ped_crossing':[],'boundary':[],'centerline':[]}
        # import ipdb;ipdb.set_trace() 
        gt_lines_instance = gt_bboxes_3d[0].instance_list
        # import pdb;pdb.set_trace()
        for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
            if gt_label_3d == 0:
                gt_dict['divider'].append(np.array(list(gt_line_instance.coords)))
            elif gt_label_3d == 1:
                gt_dict['ped_crossing'].append(np.array(list(gt_line_instance.coords)))
            elif gt_label_3d == 2:
                gt_dict['boundary'].append(np.array(list(gt_line_instance.coords)))
            elif gt_label_3d == 3:
                gt_dict['centerline'].append(np.array(list(gt_line_instance.coords)))
            else:
                raise NotImplementedError
        sample_dict['gt_map'] = gt_dict

        result_dict = result[0]['pts_bbox']
        sample_dict['pred_map'] = result_dict

        # visualize gt
        plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(15, -15)
        plt.axis('off')
        gt_centerlines = []
        for pts in gt_dict['divider']:
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.plot(x, y, color='orange',linewidth=1,alpha=0.8,zorder=-1)

        for pts in gt_dict['ped_crossing']:
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.plot(x, y, color='blue',linewidth=1,alpha=0.8,zorder=-1)

        for pts in gt_dict['boundary']:
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.plot(x, y, color='red',linewidth=1,alpha=0.8,zorder=-1)

        for pts in gt_dict['centerline']:
            instance = LineString(pts).simplify(0.2, preserve_topology=True) 
            pts = np.array(list(instance.coords))
            gt_centerlines.append(pts)
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='green',headwidth=5,headlength=6,width=0.006,alpha=0.8,zorder=-1)
        #plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
        gt_map_path = osp.join(sample_dir, 'GT_MAP.png')
        if show2:
            plt.show()
        elif restore:
            plt.savefig(gt_map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()
        
        # visualize pred
        scores_3d = result_dict['scores_3d']
        labels_3d = result_dict['labels_3d']
        pts_3d = result_dict['pts_3d']
        keep = scores_3d > args.score_thresh

        plt.figure(figsize=(4, 2))
        plt.xlim(-30, 30)
        plt.ylim(15, -15)
        plt.axis('off')
        pred_centerlines=[]
        pred_anno = {'divider':[],'ped_crossing':[],'boundary':[],'centerline':[]}
        class_by_index=['divider','ped_crossing','boundary']
        for pred_score_3d,  pred_label_3d, pred_pts_3d in zip(scores_3d[keep], labels_3d[keep], pts_3d[keep]):
            if pred_label_3d == 3:
                instance = LineString(pred_pts_3d.numpy()).simplify(0.2, preserve_topology=True)
                pts = np.array(list(instance.coords))
                pred_anno['centerline'].append(pts)
                pred_centerlines.append(pts)
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color='green',headwidth=5,headlength=6,width=0.006,alpha=0.8,zorder=-1)
            else: 
                pred_pts_3d = pred_pts_3d.numpy()
                pred_anno[class_by_index[pred_label_3d]].append(pred_pts_3d)
                pts_x = pred_pts_3d[:,0]
                pts_y = pred_pts_3d[:,1]
                plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
        #         plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)

        #plt.imshow(car_img, extent=[-1.5, 1.5, -1.2, 1.2])
        map_path = osp.join(sample_dir, 'PRED_MAP.png')
        if show2:
            plt.show()
        elif restore:
            plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        vec = dict()
        for i in range(1):
            vec[i] = dict()
            for j in range(3):
                vec[i][j] = []
        num_class = 3
        res_pts = pts_3d.unsqueeze(0)  # 1 50 20 2gt_lines_fixed_num_pts.unsqueeze(0)#
        # res_pts[:, :, :, [0,1]] = -res_pts[:, :, :, [1,0]]
        # res_pts[:,:,:,1] = -res_pts[:,:,:,1]#1是y
        labels_3d = labels_3d.unsqueeze(0)  # 1 50gt_labels_3d[0].unsqueeze(0)
        scores_3d = scores_3d.unsqueeze(0)
        labels_3d[scores_3d < 0.4] = -1
        cls_1 = torch.where(labels_3d[:] == 0)  # x
        if len(cls_1) != 0:
            vec = trans(cls_1[0], cls_1[1], res_pts, vec, name=0)
        cls_2 = torch.where(labels_3d[:] == 1)  # x y
        if len(cls_2) != 0:
            vec = trans(cls_2[0], cls_2[1], res_pts, vec, name=1)
        cls_3 = torch.where(labels_3d[:] == 2)
        if len(cls_3) != 0:
            vec = trans(cls_3[0], cls_3[1], res_pts, vec, name=2)
        all_map = []
        for i in range(1):
            single_map = []
            for j in range(3):
                if len(vec[i][j]) != 0:
                    map_mask = mask_line(vec[i][j])  # 200 100
                else:
                    map_mask = np.zeros((100, 200))
                single_map.append(map_mask)
            single_map = torch.tensor(single_map)
            t = single_map != 0
            single_map = torch.cat([(~torch.any(t, axis=0)).unsqueeze(0), single_map])  # 4 200 100
            all_map.append(single_map)
        all_map = torch.stack(all_map)  # b  4 200 100
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(all_map[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(all_map[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(all_map[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        map_path = osp.join(sample_dir, 'pred_semantic.png')
        # semantic_gt = torch.stack(semantic_gt)
        # intersects,union = get_batch_iou(all_map[:,1:],semantic_gt[:,1:])
        # total_intersects += intersects
        # total_union += union
        if show:
            plt.show()
        elif restore:
            plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)

        ##########展示预测的语义地图融合
        now_rot = img_metas[0]['can_bus'][-2]  # ego to global
        de = img_metas[0]['can_bus'][:3]  # ego to global
        #print(de)
        if pre_semantic is None:
            pre_semantic = semantic_gt#all_map



        delta_translation_global = torch.tensor(
            [de[0], de[1]]).unsqueeze(1)  # .permute(1, 0)#2 1
        curr_transform_matrix = torch.tensor([[np.cos(-now_rot), -np.sin(-now_rot)],
                                              [np.sin(-now_rot), np.cos(-now_rot)]])  # (2, 2) 世界坐标转换为当前帧自车坐标

        delta_translation_curr = torch.matmul(
            curr_transform_matrix,
            delta_translation_global).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
        delta_translation_curr = delta_translation_curr // 0.3  # (bs, 2) = (bs, 2) / (1, 2) (格子数)
        delta_translation_curr = delta_translation_curr.round().tolist()

        theta = img_metas[0]['can_bus'][-1]
        tran = [-delta_translation_curr[0], delta_translation_curr[1]]  # nuscenes是x为h，y为w
        pre_warp = affine(pre_semantic[0],  # 必须是(c, h, w)形式的tensor
                          angle=-theta,
                          # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                          translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                          scale=1,  # 浮点数，中心缩放尺度
                          shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                          interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                          fill=[0.0])  # 先旋转再平移
        pre_warp = rearrange(pre_warp, 'c h w -> h w c')
        now = rearrange(all_map[0], 'c h w -> h w c')
        pre_warp = pre_warp.long()
        now = np.array(now.long())
        pre_warp = np.array(pre_warp)
        # merge = cv2.addWeighted(now, 0.5, pre_warp, 0.5, 0)
        merge = now + pre_warp
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(merge[:, :, 1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        # merge = torch.tensor(merge)#h w c
        # merge = rearrange(merge,'h w c -> c h w')
        # intersects, union = get_batch_iou(merge.unsqueeze(0)[:,1:], semantic_gt[:, 1:])
        # total_intersects_merge += intersects
        # total_union_merge += union
        semantic_path = osp.join(sample_dir, 'semantic_merge.png')
        if show:
            plt.show()
        elif restore:
            plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)
        pre_semantic =all_map# semantic_gt#

        # rendered_cams_dict = {}
        # for key, cam_dict in img_path_dict.items():
        #     cam_img = cv2.imread(osp.join(data_path_prefix,cam_dict['filepath']))
        #     render_anno_on_pv(cam_img,pred_anno,cam_dict['lidar2img'])
        #     if 'front' not in key:
        # #         cam_img = cam_img[:,::-1,:]
        #         cam_img = cv2.flip(cam_img, 1)
        #     lw = 8
        #     tf = max(lw - 1, 1)
        #     w, h = cv2.getTextSize(caption_by_cam[key], 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        #     p1 = (0,0)
        #     p2 = (w,h+3)
        #     color=(0, 0, 0)
        #     txt_color=(255, 255, 255)
        #     cv2.rectangle(cam_img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        #     cv2.putText(cam_img,
        #                 caption_by_cam[key], (p1[0], p1[1] + h + 2),
        #                 0,
        #                 lw / 3,
        #                 txt_color,
        #                 thickness=tf,
        #                 lineType=cv2.LINE_AA)
        #     rendered_cams_dict[key] = cam_img
        #
        # new_image_height = 2048
        # new_image_width = 1550+2048*2
        # color = (255,255,255)
        # first_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        # first_row_canvas[(2048-1550):, :2048,:] = rendered_cams_dict['ring_front_left']
        # first_row_canvas[:,2048:(2048+1550),:] = rendered_cams_dict['ring_front_center']
        # first_row_canvas[(2048-1550):,3598:,:] = rendered_cams_dict['ring_front_right']
        #
        # new_image_height = 1550
        # new_image_width = 2048*4
        # color = (255,255,255)
        # second_row_canvas = np.full((new_image_height,new_image_width, 3), color, dtype=np.uint8)
        # second_row_canvas[:,:2048,:] = rendered_cams_dict['ring_side_left']
        # second_row_canvas[:,2048:4096,:] = rendered_cams_dict['ring_rear_left']
        # second_row_canvas[:,4096:6144,:] = rendered_cams_dict['ring_rear_right']
        # second_row_canvas[:,6144:,:] = rendered_cams_dict['ring_side_right']
        #
        # resized_first_row_canvas = cv2.resize(first_row_canvas,(8192,2972))
        # full_canvas = np.full((2972+1550,8192,3),color,dtype=np.uint8)
        # full_canvas[:2972,:,:] = resized_first_row_canvas
        # full_canvas[2972:,:,:] = second_row_canvas
        # cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
        # cv2.imwrite(cams_img_path, full_canvas,[cv2.IMWRITE_JPEG_QUALITY, 70])
        # plt.figure()
        # plt.axis('off')  # 去坐标轴
        # plt.xticks([])  # 去 x 轴刻度
        # plt.yticks([])  # 去 y 轴刻度
        # plt.imshow(full_canvas)
        # plt.show()

        #final_dict[pts_filename] = sample_dict
        prog_bar.update()

    #mmcv.dump(final_dict, osp.join(args.show_dir, 'final_dict.pkl'))
    logger.info('\n DONE vis test dataset samples gt label & pred')
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
    #coords = np.sort(coords, axis=1)
    cv2.polylines(map_mask, [coords], False, color=idx, thickness=5)
    return map_mask,idx
def mask_line(layer_geom, local_box=(0,0,30,60),canvas_size=(100,200),idx=1):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)  # (-30,-15,30,15)

    canvas_h = canvas_size[0]  # 400
    canvas_w = canvas_size[1]  # 200
    scale_height = canvas_h / patch_h  # 200/60
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0#
    trans_y = -patch_y + patch_h / 2.0#30

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
        if name != 1:
            vec[h[i]][name].append(LineString(t0))
        else:
            vec[h[i]][name].append(LinearRing(t0))

    return vec
if __name__ == '__main__':
    main()
