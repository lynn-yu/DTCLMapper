import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
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
import cv2
from einops import rearrange
from shapely.geometry import LineString, box,LinearRing
from shapely import affinity
import matplotlib.pyplot as plt
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
from scipy.spatial import Voronoi, voronoi_plot_2d
import open3d as o3d
CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]
# we choose these samples not because it is easy but because it is hard
CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('--config',
                        default="",
                        help='test config file path')
    parser.add_argument('--checkpoint',
                        default='',
                        )
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
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
        args.show_dir = osp.join('./work_dirs_ori',
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
    car_img = Image.open('/figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')



    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    show=True
    restore=False
    pre_semantic = None
    pre_gt=None
    total_intersects = 0
    total_union = 0
    total_intersects_merge = 0
    total_union_merge = 0
    dis_num = torch.zeros([1])
    for i, data in enumerate(data_loader):

        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue

        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]
        #data['img'][0].data[0] = torch.zeros_like(data['img'][0].data[0])



        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        # import pdb;pdb.set_trace()
        # if pts_filename not in CANDIDATE:
        #     continue

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        ########### semantic_gt展示gt语义
        #semantic_gt = data['semantic_gt'].data[0]
        #semantic_gt = rearrange(semantic_gt, 'b c h w  -> b h w c')
        # plt.figure(figsize=(1, 2), dpi=100)
        # plt.axis('off')  # 去坐标轴
        # plt.xticks([])  # 去 x 轴刻度
        # plt.yticks([])  # 去 y 轴刻度
        # plt.imshow(semantic_gt[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        # plt.imshow(semantic_gt[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        # plt.imshow(semantic_gt[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        # semantic_path = osp.join(sample_dir, 'semantic_gt.png')
        # if show:
        #     plt.show()
        # else:
        #     plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)


        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        for filepath in filename_list:
            filename = osp.basename(filepath)
            filename_splits = filename.split('__')
            # sample_dir = filename_splits[0]
            # sample_dir = osp.join(args.show_dir, sample_dir)
            # mmcv.mkdir_or_exist(osp.abspath(sample_dir))
            img_name = filename_splits[1] + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            # img_path_list.append(img_path)
            shutil.copyfile(filepath,img_path)
            img_path_dict[filename_splits[1]] = img_path


        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_1_list.append(cam_img)
        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_2_list.append(cam_img)
        row_1_img=cv2.hconcat(row_1_list)
        row_2_img=cv2.hconcat(row_2_list)
        cams_img = cv2.vconcat([row_1_img,row_2_img])
        cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
        #cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])

        for vis_format in args.gt_format:
            if vis_format == 'se_pts':
                gt_line_points = gt_bboxes_3d[0].start_end_points
                for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1,2).numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
            elif vis_format == 'bbox':
                gt_lines_bbox = gt_bboxes_3d[0].bbox
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                    gt_bbox_3d = gt_bbox_3d.numpy()
                    xy = (gt_bbox_3d[0],gt_bbox_3d[1])
                    width = gt_bbox_3d[2] - gt_bbox_3d[0]
                    height = gt_bbox_3d[3] - gt_bbox_3d[1]
                    # import pdb;pdb.set_trace()
                    plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[gt_label_3d],facecolor='none'))
                    # plt.Rectangle(xy, width, height,color=colors_plt[gt_label_3d])
                # continue
            elif vis_format == 'fixed_num_pts':

                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points


                ###########展示gt raster
                vec = dict()
                for ii in range(1):
                    vec[ii] = dict()
                    for j in range(3):
                        vec[ii][j] = []
                res_pts = gt_lines_fixed_num_pts.unsqueeze(0)
                labels_3d = gt_labels_3d[0].unsqueeze(0)
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
                for ii in range(1):
                    single_map = []
                    for j in range(3):
                        if len(vec[ii][j]) != 0:
                            map_mask = mask_line(vec[ii][j])  # 200 100
                        else:
                            map_mask = np.zeros((200, 100))
                        single_map.append(map_mask)
                    single_map = torch.tensor(single_map)
                    t = single_map != 0
                    single_map = torch.cat([(~torch.any(t, axis=0)).unsqueeze(0), single_map])  # 4 200 100
                    all_map.append(single_map)
                gt_map = torch.stack(all_map)  # b  4 200 100
                plt.figure()
                plt.axis('off')  # 去坐标轴
                plt.xticks([])  # 去 x 轴刻度
                plt.yticks([])  # 去 y 轴刻度
                plt.imshow(gt_map[0][1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
                plt.imshow(gt_map[0][2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
                plt.imshow(gt_map[0][3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
                map_path = osp.join(sample_dir, 'gt_semantic.png')
                if show:
                    plt.show()
                elif restore:
                    plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
                plt.close()

                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[4], pc_range[1])
                plt.axis('off')
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):


                    # import pdb;pdb.set_trace()
                    pts = gt_bbox_3d.numpy()

                    x = np.array([pt[0] for pt in pts])#横向
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])


                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')#
                #展示矢量地图
                if show:
                    plt.show()
                elif restore:
                    plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png', dpi=1200)
                plt.close()
            elif vis_format == 'polyline_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])

                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_pts_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()

            else:
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')



        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']# 4 50 3
        pts_3d = result_dic['pts_3d']



        keep = scores_3d > args.score_thresh

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[4], pc_range[1])
        plt.axis('off')
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):

            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]


            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)


            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)



        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        ##############展示预测矢量地图
        if show:
            plt.show()
        elif restore :
            plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()
        #################展示矢量语义地图
        vec = dict()
        for ii in range(1):
            vec[ii] = dict()
            for j in range(3):
                vec[ii][j] = []
        num_class = 3
        res_pts = pts_3d.unsqueeze(0)#1 50 20 2gt_lines_fixed_num_pts.unsqueeze(0)#
        #res_pts[:, :, :, [0,1]] = -res_pts[:, :, :, [1,0]]
        #res_pts[:,:,:,1] = -res_pts[:,:,:,1]#1是y
        labels_3d = labels_3d.unsqueeze(0)#1 50gt_labels_3d[0].unsqueeze(0)
        scores_3d = scores_3d.unsqueeze(0)
        labels_3d[scores_3d<0.4] = -1
        cls_1 = torch.where(labels_3d[:] == 0)#x
        if len(cls_1) != 0:
            vec = trans(cls_1[0], cls_1[1], res_pts, vec, name=0)
        cls_2 = torch.where(labels_3d[:] == 1)  # x y
        if len(cls_2) != 0:
            vec = trans(cls_2[0], cls_2[1], res_pts, vec, name=1)
        cls_3 = torch.where(labels_3d[:] == 2)
        if len(cls_3) != 0:
            vec = trans(cls_3[0], cls_3[1], res_pts, vec, name=2)
        all_map = []
        for ii in range(1):
            single_map = []
            for j in range(3):
                if len(vec[ii][j]) != 0:
                    map_mask = mask_line(vec[ii][j])  # 200 100
                else:
                    map_mask = np.zeros((200, 100))
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

        if show:
            plt.show()
        elif restore:
            plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()
        ##########展示预测的语义地图融合
        now_rot = img_metas[0]['can_bus'][-2]#ego to global
        de = img_metas[0]['can_bus'][:3]#ego to global
        if pre_semantic is None:
            pre_semantic =  all_map
        if pre_gt is None:
            pre_gt = gt_map


        delta_translation_global = torch.tensor(
            [de[0], de[1]]).unsqueeze(1)#.permute(1, 0)#2 1
        curr_transform_matrix = torch.tensor([[np.cos(-now_rot), -np.sin(-now_rot)],
                                          [np.sin(-now_rot), np.cos(-now_rot)]])  # (2, 2) 世界坐标转换为当前帧自车坐标

        delta_translation_curr = torch.matmul(
            curr_transform_matrix,
            delta_translation_global).squeeze()  # (bs, 2, 1) -> (bs, 2) = (bs, 2, 2）* (bs, 2, 1) (真实坐标)
        delta_translation_curr=delta_translation_curr//0.3   # (bs, 2) = (bs, 2) / (1, 2) (格子数)
        delta_translation_curr = delta_translation_curr.round().tolist()

        theta = img_metas[0]['can_bus'][-1]
        tran = [-delta_translation_curr[1], -delta_translation_curr[0]]#nuscenes是x为h，y为w
        pre_warp = affine(pre_semantic[0],  # 必须是(c, h, w)形式的tensor
                          angle=360-theta,  # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                          translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                          scale=1,  # 浮点数，中心缩放尺度
                          shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                          interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                          fill=[0.0])#先旋转再平移
        pre_warp_gt = affine(pre_gt[0],  # 必须是(c, h, w)形式的tensor
                          angle=360 - theta,
                          # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                          translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                          scale=1,  # 浮点数，中心缩放尺度
                          shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                          interpolation=InterpolationMode.BILINEAR,  # 二维线性差值，默认是最近邻差值
                          fill=[0.0])  # 先旋转再平移
        pre_warp = rearrange(pre_warp, 'c h w -> h w c')
        now = rearrange(  all_map[0], 'c h w -> h w c')


        pre_warp = pre_warp.long()
        now = np.array(now.long())
        pre_warp = np.array(pre_warp)
        merge = np.abs(pre_warp-now)
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(merge[:, :, 1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)

        semantic_path = osp.join(sample_dir, 'semantic_merge.png')
        if show:
            plt.show()
        elif restore:
            plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)
        pre_warp_gt = rearrange(pre_warp_gt, 'c h w -> h w c')
        now_gt = rearrange(gt_map[0], 'c h w -> h w c')
        pre_warp_gt = pre_warp_gt.long()
        now_gt = np.array(now_gt.long())
        pre_warp_gt = np.array(pre_warp_gt)

        merge = np.abs(pre_warp_gt - now_gt)
        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(merge[:, :, 1], vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 2], vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(merge[:, :, 3], vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        semantic_path = osp.join(sample_dir, 'gt_semantic_merge.png')
        if show:
            plt.show()
        elif restore:
            plt.savefig(semantic_path, bbox_inches='tight', format='png', dpi=1200)
        pre_semantic = all_map
        pre_gt = gt_map
        prog_bar.update()
    print(666,dis_num/81)
    logger.info('\n DONE vis test dataset samples gt label & pred')
def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[1]):
            pred = pred_map[:, i]
            tgt = gt_map[:, i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
        intersects,union = torch.tensor(intersects), torch.tensor(unions)

    return intersects,union
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
def mask_line(layer_geom, local_box=(0,0,60,30),canvas_size=(200,100),idx=1):
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
    # to = np.array([[1,2],[2,2],[3,3],[4,4]])
    # l = LineString(t0)


