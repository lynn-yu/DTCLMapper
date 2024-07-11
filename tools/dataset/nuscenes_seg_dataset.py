import os
import os.path as osp
from functools import reduce

import mmcv
import numpy as np


from torch.utils.data import Dataset



import json
import cv2
import numpy as np
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from einops import rearrange
from pathlib import Path
from vector_map import VectorizedLocalMap
from rasterize import preprocess_map
import torch.nn as nn
from skimage.util import random_noise
import random

import pyquaternion
import torch.nn.functional as F
import math
import imutils
import time
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from torchvision.transforms import InterpolationMode
from shapely.geometry import LineString

INTERPOLATION = cv2.LINE_8

MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

NUM_CLASSES = 3
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600
class Point_color:
    def __init__(self, point,color):
        self.point=point
        self.color = color
def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot
def get_batch_iou(pred_map, gt_map):
    intersects = []
    unions = []
    with torch.no_grad():
        pred_map = pred_map.bool()
        gt_map = gt_map.bool()

        for i in range(pred_map.shape[0]):
            pred = pred_map[i]
            tgt = gt_map[ i]
            intersect = (pred & tgt).sum().float()
            union = (pred | tgt).sum().float()
            intersects.append(intersect)
            unions.append(union)
    return torch.tensor(intersects), torch.tensor(unions)
class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))#对比度增强
normalize_ipm = torchvision.transforms.Compose((
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))#对比度增强
nor_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))
denor_img = torchvision.transforms.Compose((
    torchvision.transforms.ToPILImage(),
))
denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
                                                 ))
def get_transformation_matrix(R, t, inv=False):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R if not inv else R.T
    pose[:3, -1] = t if not inv else R.T @ -t

    return pose
def perspective_2(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def get_pose(rotation, translation, inv=False, flat=False):
    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)

    return get_transformation_matrix(R, t, inv=inv)
def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0

    return inverse_matrix
class NuScenesDataset(Dataset):
    CLASSES = (
        'other','Road', 'Ped', 'Div',)
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230],]
    def __init__(self,version, dataroot, data_conf, is_train,classes=None,palette=None,test_mode=False,seg_file=None,ignore_index=255,reduce_zero_label=False,pre=False ):
        self.data_root = dataroot
        self.test_mode = test_mode
        self.load_interval=1
        self.ignore_index = ignore_index
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        self.seg_file = seg_file
        self.reduce_zero_label = reduce_zero_label

        self.xbound = data_conf['xbound']
        self.ybound = data_conf['ybound']
        self.is_train = is_train
        self.data_conf = data_conf
        x = self.xbound[1] - self.xbound[0]
        y = self.ybound[1] - self.ybound[0]
        self.patch_size = (x, y)  # 60 30
        canvasx = (self.xbound[1] - self.xbound[0]) / self.xbound[2]  # 400
        canvasy = (self.ybound[1] - self.ybound[0]) / self.ybound[2]  # 200
        self.canvas_size = (int(canvasx), int(canvasy))  # 512 512
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()

        # self.plane = plane_grid(self.xbound,self.ybound,0.5)
        self.h = data_conf['h']
        self.w = data_conf['w']
        self.len = 4
        self.indices = self.get_indices()
        self.pre=pre
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos
    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette
    def get_indices(self):
        indices = []
        for index in range(len(self.samples)):  # 所有sample集合
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.len):  # 3
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.samples):
                    is_valid_data = False
                    break
                rec = self.samples[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):  # 如果不属于同一个场景
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]
        if version == 'v1.0-mini':
            self.ipm_path = "/data/lsy/dataset/nuScenes/v1.0/ipm_mini"
        else:
            self.ipm_path = "/data/lsy/dataset/nuScenes/v1.0/ipm_tra"
        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples#
    def __len__(self):
        """Total number of samples of data."""
        return len(self.indices)#len(self.data_infos)#


    def __getitem__(self, idx):

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)
    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        car_trans = ego_pose['translation']  # 车辆到世界坐标系的变换
        pos_rotation = Quaternion(ego_pose['rotation'])
        rot_t = torch.tensor(pyquaternion.Quaternion(ego_pose['rotation']).rotation_matrix)
        tran_t = torch.tensor(car_trans)
        Rs = torch.eye(4, device=rot_t.device).view(4, 4)
        Rs[:3, :3] = rot_t
        Rs[:3, 3] = tran_t
        RTs = Rs
        patch_angle = quaternion_yaw(pos_rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        rot = patch_angle / 180 * np.pi
        return torch.tensor(car_trans), patch_angle, RTs

    def get_cam(self,fx,fy,image, R,T, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,):
        res={}
        res['FoVx'] = fx
        res['FoVy'] = fy
        res['original_image'] = image.clamp(0.0, 1.0)
        res['image_width'] = image.shape[2]
        res['image_height'] = image.shape[1]
        zfar = 32.0
        znear = 0.01
        res['trans'] = trans
        res['scale'] = scale
        W2V_RT = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0,1)
        res['world_view_transform'] = W2V_RT
        projection_matrix= getProjectionMatrix(znear=znear, zfar=zfar, fovX=fx,fovY=fy).transpose(0, 1)
        res['projection_matrix'] =projection_matrix
        res['full_proj_transform'] = (W2V_RT.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        res['camera_center'] = W2V_RT.inverse()[3, :3]
        return res

    def get_imgs_full(self, rec):#rec不同时间帧数据
        imgs = []
        trans = []
        rots = []
        intrins = []

        sd_rec = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        lidar2img_rts = []
        cam_list= []
        id=0
        for cam in CAMS:#不同相机位置名字
            samp = self.nusc.get('sample_data', rec['data'][cam])  # 获得本相机的图片数据
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])  # 图片存放路径
            # resize, resize_dims = self.sample_augmentation()  # 图片变换尺度，变换大小，目标数据尺寸
            img = Image.open(imgname)
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims)  # 输出3*3旋转和平移矩阵
            # img = normalize_img(img)  # 归一化3 128 325
            img = nor_img(img)  # 显示原图
            # img = np.array(img).astype('uint8')
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])  # 传感器标定数据
            trans.append(torch.Tensor(sens['translation']))  # 标定位移变换，相机到车辆的变换
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))  # 标定旋转变换
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
            imgs.append(img)  # 不同角度相机累积

            sd_rec = self.nusc.get('sample_data', rec['data'][cam])
            cs_record = self.nusc.get('calibrated_sensor',sd_rec['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])
            l2e_r_s = cs_record['rotation']
            l2e_t_s = cs_record['translation']
            e2g_r_s = pose_record['rotation']
            e2g_t_s = pose_record['translation']

            l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
            R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

            lidar2cam_r = np.linalg.inv(R.T)
            lidar2cam_t = T @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_rt_t = lidar2cam_rt.T

            intrinsic = torch.Tensor(sens['camera_intrinsic'])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt_t)
            lidar2img_rts.append(torch.tensor(lidar2img_rt))
            intrinsic = np.array(sens['camera_intrinsic'])
            fx = intrinsic[0,0]
            fy = intrinsic[1,1]
            # Rs = torch.eye(4).view(4, 4)
            # Rs[:3, :3] = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            # Rs[:3, 3] = torch.Tensor(sens['translation'])
            # W2VRT = self.invers_matrix(Rs)
            # R = np.transpose(Quaternion(sens['rotation']).rotation_matrix)
            # T = np.array(sens['translation'])
            # gt_img = img/255.0
            # cam = self.get_cam(fx,fy,gt_img,R,T)
            # cam_list.append(cam)
            # id +=1



        imgs = torch.stack(imgs)#6 c h w
        trans = torch.stack(trans)
        rots = torch.stack(rots)
        intrins = torch.stack(intrins)
        lidar2img_rts = torch.stack( lidar2img_rts)
        return imgs,rots,trans,intrins,lidar2img_rts,imgs

    def ipm_to_world(self,rots, trans, intrix, imgs,n2p_ego=None,h=0):
        Rs = torch.eye(4, device=rots.device).view(1, 4, 4).repeat(6, 1, 1)
        rots = torch.transpose(rots, dim0=1, dim1=2)
        Rs[:, :3, :3] = rots
        Ts = torch.eye(4, device=trans.device).view(1, 4, 4).repeat(6, 1, 1)
        Ts[:, :3, 3] = -trans  # -trans
        RTs = Rs @ Ts

        xmin, xmax = -30, 30
        num_x = self.h
        ymin, ymax = -15, 15  # 2, 20
        num_y = self.w
        ipm = []
        pix_cord = []
        for i in range(6):
            x = torch.linspace(xmin, xmax, num_x)
            y = torch.linspace(ymax, ymin, num_y)

            x, y = torch.meshgrid(x, y)

            x = x.flatten()#.cuda()  # 800*400
            y = y.flatten()#.cuda()   #
            z = torch.ones_like(x)*h
            d = torch.ones_like(x)
            coords = torch.stack([x, y, z, d], axis=0)  # 4 800*400
            pose = coords
            if n2p_ego is not None:
                coords = n2p_ego@coords
            cam_coords = RTs[i] @ coords
            cam_coords = cam_coords[:3]  # 3 320000
            mask1 = cam_coords[0] > -35
            mask2 = cam_coords[0] < 35
            mask3 = cam_coords[2] > 0
            mask4 = cam_coords[2] < 50
            mask = mask1 & mask2
            mask = mask & mask3
            mask_c = mask & mask4
            zc = cam_coords[2]  # 1 320000
            pixel_coords = intrix[i] @ cam_coords  #
            pixel_coords = torch.div(pixel_coords, zc)  # 3 320000
            mask1 = pixel_coords[0] > -0.999
            mask2 = pixel_coords[0] < 1600.01
            mask3 = pixel_coords[1] > -0.999
            mask4 = pixel_coords[1] < 900.01
            mask = mask1 & mask2
            mask = mask & mask3
            mask_p = mask & mask4
            pixel_coords = pixel_coords.permute(1, 0).contiguous()  # 320000 3
            pixel_coords = pixel_coords.view(num_x, num_y, -1)  # 400 800 3
            pixel_coords = pixel_coords[:, :, :2]
            img = imgs[i]
            img_c, img_h, img_w = img.shape  # 3 900 1600
            img = img.permute(1, 2, 0).contiguous()  # 900 1600 3
            pix_x, pix_y = torch.split(pixel_coords, 1, dim=-1)  #
            pix_x0 = torch.floor(pix_x)#400 200 1
            pix_y0 = torch.floor(pix_y)
            y_max = (img_h - 1)
            x_max = (img_w - 1)
            pix_x0 = torch.clip(pix_x0, 0, x_max)
            pix_y0 = torch.clip(pix_y0, 0, y_max)
            dim = img_w
            base_y0 = pix_y0 * dim
            idx00 = (pix_x0 + base_y0).view(-1, 1).repeat(1, img_c).long()  # 320000 3
            imgs_flat = img.reshape([-1, img_c])
            im00 = torch.gather(imgs_flat, 0, idx00)  ##320000 3
            im00 = im00.view(num_x, num_y, -1)  # 400 200 3
            mask = mask_p & mask_c
            mask = mask.view(num_x, num_y)  # 400 200
            mask = mask.unsqueeze(2).repeat(1, 1, img_c)
            im = im00 * mask  # 400 200 3
            # pix_x0 = pix_x0-img_w//2
            # pix_y0 = pix_y0 - img_h // 2
            pix_z0 = torch.ones_like(pix_x0)*(i+1)
            pix = torch.cat((pix_x0,pix_y0,pix_z0),dim=-1)#400 200 3
            pix = pix*mask

            ipm.append(im)
            pix_cord.append(pix)
        ipm = torch.stack(ipm)  # 6 400 200 3
        ipm = torch.sum(ipm, dim=0)# 400 200 3
        pix_cord = torch.stack(pix_cord)#6 400 200 3
        pix_cord = torch.sum(pix_cord, dim=0)  # 400 200 3


        return ipm,pose


    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']#此场景日志信息，采集地名字
        #ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])  # 车本身在世界坐标系位姿
        sd_rec = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor',sd_rec['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])#车本身在世界坐标系位姿
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        lidar2ego[:3, 3] = cs_record['translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego_pose['rotation']).rotation_matrix
        ego2global[:3, 3] = ego_pose['translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        vectors = self.vector_map.gen_vectorized_samples(location,lidar2global_translation, lidar2global_rotation  )#lidar2global_translation, lidar2global_rotation

        #ex_vec,in_vec = self.vector_map.static_layer(location,lidar2global_translation, lidar2global_rotation,['lane', 'road_segment'])
        #dict[road_divide:  ,lane_divied:list[5],5个对象均分点, ped_crpssiing:list[]]

        return vectors,None,None

    def get_vec_semantic_map(self,lidar2global,location):

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)
        vectors = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        semantic_mask, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, 3,
                                                                      self.thickness, self.angle_class)

        semantic_masks = semantic_mask != 0
        semantic_iou = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])

        num_cls = semantic_masks.shape[0]
        indices = torch.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = torch.sum(semantic_masks * indices, dim=0)
        # semantic_indices = semantic_indices.unsqueeze(0)

        semantic_masks = 255 * semantic_masks
        mask = semantic_mask == 0
        mask = 1 * mask
        mask_a = torch.sum(mask, dim=0)
        mask_a = mask_a == 3
        mask_a = mask_a.unsqueeze(0).repeat(3, 1, 1)
        mask = mask_a * 255
        semantic_masks += mask

        return semantic_masks, semantic_indices, semantic_iou

    def get_semantic_map(self, rec):
        vectors,ex_vec,in_vec = self.get_vectors(rec)
        semantic_mask, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, 3, self.thickness, self.angle_class)


        semantic_masks = semantic_mask != 0
        vi = semantic_masks.numpy()
        semantic_iou = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])

        num_cls = semantic_masks.shape[0]
        indices = torch.arange(1, num_cls + 1).reshape(-1, 1, 1)
        semantic_indices = torch.sum(semantic_masks  * indices, dim=0)
        #semantic_indices = semantic_indices.unsqueeze(0)

        semantic_masks = 255 * semantic_masks
        mask = semantic_mask == 0
        mask = 1 * mask
        mask_a = torch.sum(mask, dim=0)
        mask_a = mask_a == 3
        mask_a = mask_a.unsqueeze(0).repeat(3, 1, 1)
        mask = mask_a * 255
        semantic_masks += mask
        #gt_pv_semantic_mask = self.get_pv_mask(ex_vec,in_vec,lidar2img)

        return semantic_masks,semantic_indices,semantic_iou,semantic_iou,#gt_pv_semantic_mask

    def get_pv_mask(self,ex_vec,in_vec,lidar2feat):
        gt_pv_semantic_mask = np.zeros((6, 1, 900, 1600), dtype=np.uint8)

        for i in range(len(ex_vec)):
            ex = ex_vec[i]
            flag=False
            if len(in_vec[i][0])>0:
                inter = in_vec[i][0]
                flag=True
            for cam_index in range(6):
                if len(ex)>0:
                    self.line_ego_to_pvmask(ex.numpy(), gt_pv_semantic_mask[cam_index][0], lidar2feat[cam_index],color=1, thickness=1)
                if flag:
                    self.line_ego_to_pvmask(inter.numpy(), gt_pv_semantic_mask[cam_index][0], lidar2feat[cam_index], color=0, thickness=1)
        return gt_pv_semantic_mask

    def line_ego_to_pvmask(self,
                          line_ego,
                          mask,
                          lidar2feat,
                          color=1,
                          thickness=1,
                          z=-1.6):
        #distances = np.linspace(0, line_ego.length, 200)
        coords = line_ego#np.array([list(line_ego.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        pts_num = len(coords)
        zeros = np.zeros((pts_num,1))
        zeros[:] = z
        ones = np.ones((pts_num,1))
        lidar_coords = np.concatenate([coords,zeros,ones], axis=1).transpose(1,0)
        pix_coords = perspective_2(lidar_coords, lidar2feat)
        a,b = pix_coords.shape
        if a!=0 and b!=0:

            pix_coords=pix_coords.unsqueeze(0)
            coords = np.asarray(pix_coords, np.int32)
            cv2.fillPoly(mask, coords, color, INTERPOLATION)
        #cv2.polylines(mask, coords, False, color=color, thickness=thickness)
    def prev_to_now(self,pre_rec,now_rec,pre_ipm):
        trans_n,rot_n,rot_np=self.get_ego_pose(now_rec)
        trans_p, rot_p, rot_pp = self.get_ego_pose(pre_rec)

        delta = trans_n-trans_p
        delta_translation_global = delta[:2].unsqueeze(-1)#2 1

        curr_transform_matrix = torch.tensor([[np.cos(-rot_n), -np.sin(-rot_n)],
                                          [np.sin(-rot_n), np.cos(-rot_n)]])#2 2
        delta_translation_curr = torch.matmul(
            curr_transform_matrix.to(torch.float32),
            delta_translation_global.to(torch.float32))
        solution = torch.ones_like(delta_translation_curr)*0.15
        delta_translation_curr = torch.div(delta_translation_curr,solution,rounding_mode='floor') # (bs, 2) = (bs, 2) / (1, 2) (格子数)
        tran = [-delta_translation_curr[0], delta_translation_curr[1]]
        pre_ipm = rearrange(pre_ipm,'h w c -> c h w')
        pre_warp = affine(pre_ipm,  # 必须是(c, h, w)形式的tensor
                          angle=-(rot_n-rot_p),
                          # 是角度而非弧度；如果在以左下角为原点的坐标系中考虑，则逆时针旋转为正；如果在以左上角为原点的坐标系中考虑，则顺时针旋转为正；两个坐标系下图片上下相反
                          translate=tran,  # 是整数而非浮点数，允许出现负值，是一个[x, y]的列表形式， 其中x轴对应w，y轴对应h
                          scale=1,  # 浮点数，中心缩放尺度
                          shear=0,  # 浮点数或者二维浮点数列表，切变度数，浮点数时是沿x轴切变，二维浮点数列表时先沿x轴切变，然后沿y轴切变
                          #interpolation=InterpolationMode.NEAREST,  # 二维线性差值，默认是最近邻差值
                          fill=[0.0])
        #pre_warp =  torch.tensor(pre_warp)
        pre_warp = rearrange(pre_warp, 'c h w -> h w c')
        return pre_warp

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            location=info['map_location'],
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            lidar_path=info["lidar_path"],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            map_location=info['map_location'],
            semantic_mask=[],
            instance_mask=[],
            pre2now_futuremotion=[],
            ego_motion=[],
            prev_gt_labels_3d=[],
            prev_gt_bboxes_3d=[]
        )
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        if True:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            input_dict["camego2global"] = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                cam_intrinsics.append(viewpad)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    cam_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camego to global transform
                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    cam_info['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = cam_info['ego2global_translation']
                camego2global = torch.from_numpy(camego2global)
                input_dict["camego2global"].append(camego2global)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    cam_intrinsic=cam_intrinsics,
                ))

        import pyquaternion
        e2g_trans_matrix = np.zeros((4, 4), dtype=np.float32)
        e2g_rot = input_dict['ego2global_rotation']
        e2g_trans = input_dict['ego2global_translation']
        e2g_trans_matrix[:3, :3] = pyquaternion.Quaternion(
            e2g_rot).rotation_matrix
        e2g_trans_matrix[:3, 3] = np.array(e2g_trans)
        e2g_trans_matrix[3, 3] = 1.0
        input_dict['ego_motion'] = e2g_trans_matrix

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360#确保方向么，逆时针为正
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']
        lidar2global = ego2global @ lidar2ego
        input_dict['lidar2global'] = lidar2global
        return input_dict

    def invers_matrix(self,egopose):
        inverse_matrix = torch.zeros((4, 4))
        rotation = egopose[:3, :3]
        translation = egopose[:3, 3]
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -(rotation.T@translation)
        inverse_matrix[3, 3] = 1.0
        return inverse_matrix
    def prev_to_now_RTs(self, pre_rec, now_rec):
        trans_n, rot_n, ego2world_n = self.get_ego_pose(now_rec)
        trans_p, rot_p, ego2world_p = self.get_ego_pose(pre_rec)
        world2ego_p = self.invers_matrix(ego2world_p)
        RTs =world2ego_p@ego2world_n
        RTs[3, :3] = 0.0
        RTs[3, 3] = 1.0
        return RTs

    def prepare_train_img(self, idx):
        # input_dict = self.get_data_info(idx)
        # frame_idx = input_dict['frame_idx']
        # location = input_dict['location']
        # filename = input_dict['img_filename']
        # img_list = [mmcv.imread(name, 'unchanged') for name in filename]
        # img = np.stack(img_list, axis=-1)
        # img = torch.tensor([img[..., i] for i in range(img.shape[-1])])#6 900 1600 3
        #
        # img = rearrange(img,'n h w c -> n c h w')
        # rot_tran = torch.tensor(input_dict["camera2ego"])
        # rot = rot_tran[:,:3, :3]
        # tran = rot_tran[:,:3, 3]
        # intrix = torch.tensor(input_dict['cam_intrinsic'])[:,:3, :3].to(torch.float32)
        # ipm, pix_cord = self.ipm_to_world(rot.cuda(), tran.cuda(), intrix.cuda(), img.cuda())
        # lidar2global=input_dict['lidar2global']
        # semantic_masks, semantic_indices, semantic_iou, = self.get_vec_semantic_map(lidar2global,location)
        index_now = self.indices[idx][-1]
        rec = self.samples[index_now]
        tran_pose, _, now_RTs = self.get_ego_pose(rec)
        img, rot, tran, intrix, lidar2img,cam_list = self.get_imgs_full(rec)
        color = []
        point=[]
        h=[0,-0.2,0.2]
        for i in range(len(h)):
            ipm, pose = self.ipm_to_world(rot, tran, intrix, img,h=h[i])
            color.append(ipm)
            point.append(pose)
        #ipm1, pix_cord = self.ipm_to_world(rot, tran, intrix, img, h=0.2)
        semantic_masks, semantic_indices, semantic_iou, gt_pv_semantic_mask = self.get_semantic_map(rec)

        results={}
        results['cam_img']=img
        results['iou']=semantic_iou
        results['img']=torch.stack(color)
        results['point'] = torch.stack(point)
        #results['img1'] = ipm1
        results['gt_semantic_seg'] =semantic_indices
        results['seg_fields']=['gt_semantic_seg','pre_ipm']
        results['intrix']=intrix
        a = np.zeros((6,4,4))
        a[:,:3,:3] = rot.numpy()
        a[:, :3, 3] = tran.numpy()
        a[:,3,3]=1
        results['extrix'] = a
        if self.pre:
            index = self.indices[idx][0]
            rec_pre = self.samples[index]
            _, _, pre_RTs = self.get_ego_pose(rec_pre)
            img_pre, rot_pre, tran_pre, intrix_pre, lidar2img,cam_list_pre = self.get_imgs_full(rec_pre)
            n2p_ego = self.prev_to_now_RTs(rec_pre, rec)
            ipm_pre, pix_cord = self.ipm_to_world(rot_pre, tran_pre, intrix_pre, img_pre,n2p_ego,h=0)
            #ego_motion = invert_matrix_egopose_numpy(now_RTs).dot(pre_RTs)
            ego_motion = invert_matrix_egopose_numpy(pre_RTs).dot(now_RTs)
            results['RTs'] = ego_motion
            results['pre_ipm']=ipm_pre
            #results['cam_list_pre'] = cam_list_pre
            results['cam_img_pre'] = img_pre
        else:
            results['pre_ipm'] = ipm

        return results

    def prepare_test_img(self, idx):
        # input_dict = self.get_data_info(idx)
        # frame_idx = input_dict['frame_idx']
        # scene_token = input_dict['scene_token']
        # filename = results['img_filename']
        # img_list = [mmcv.imread(name, self.color_type) for name in filename]
        # img = np.stack(img_list, axis=-1)
        # img = [img[..., i] for i in range(img.shape[-1])]
        # img = rearrange(img, 'n h w c -> n c h w')
        # rot_tran = input_dict["camera2ego"]
        # rot = rot_tran[:, :3, :3]
        # tran = rot_tran[:, :3, 3]
        # intrix = input_dict['cam_intrinsic']
        # ipm, pix_cord = self.ipm_to_world(rot.cuda(), tran.cuda(), intrix.cuda(), img.cuda())
        # lidar2global = input_dict['lidar2global']
        # semantic_masks, semantic_indices, semantic_iou, = self.get_vec_semantic_map(lidar2global)

        index_now = self.indices[idx][-1]
        rec = self.samples[index_now]
        tran_pose, _, _ = self.get_ego_pose(rec)
        img, rot, tran, intrix, lidar2img, cam_list = self.get_imgs_full(rec)
        ipm, pix_cord = self.ipm_to_world(rot, tran, intrix, img,h=-0.2)
        semantic_masks, semantic_indices, semantic_iou, gt_pv_semantic_mask = self.get_semantic_map(rec)

        results = {}
        # results['cam_list']=cam_list
        results['iou'] = semantic_iou
        results['img'] = ipm
        results['gt_semantic_seg'] = semantic_indices
        results['seg_fields'] = ['gt_semantic_seg', 'pre_ipm']

        if self.pre:
            index = self.indices[idx][0]
            rec_pre = self.samples[index]
            img_pre, rot_pre, tran_pre, intrix_pre, lidar2img, cam_list_pre = self.get_imgs_full(rec_pre)
            # rt = torch.eye(4)
            # rt = rt.unsqueeze(0).repeat(6, 1, 1)
            # rt[:, :3, :3] = rot_pre
            # rt[:, :3, 3] = tran_pre
            # pre_now = self.prev_to_now_RTs(rec_pre, rec)
            # results['img_pre'] = img_pre
            # results['ex_pre'] = pre_now @ rt
            # results['in_pre'] = intrix_pre
            # results['pre_ipm'] = ipm
            n2p_ego = self.prev_to_now_RTs(rec_pre, rec)
            ipm_pre, pix_cord = self.ipm_to_world(rot_pre, tran_pre, intrix_pre, img_pre, n2p_ego)

            results['pre_ipm'] = ipm_pre
            # results['cam_list_pre'] = cam_list_pre
            results['img_pre'] = img_pre
        else:
            # index = self.indices[idx][0]
            # rec_pre = self.samples[index]
            # img_pre, rot_pre, tran_pre, intrix_pre, lidar2img, cam_list_pre = self.get_imgs_full(rec_pre)
            # rt = torch.eye(4)
            # rt =rt.unsqueeze(0).repeat(6,1,1)
            # rt[:,:3,:3]=rot_pre
            # rt[:,:3,3]=tran_pre
            # pre_now = self.prev_to_now_RTs(rec_pre,rec)
            # results['img_pre'] = img_pre
            # results['ex_pre'] = pre_now@rt
            # results['in_pre'] = intrix_pre
            results['pre_ipm'] = ipm

        return results

    def format_results(self, results, **kwargs):
        """Place holder to format result to dataset specific output."""
        pass

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        dataset_length = len(self)
        prog_bar = mmcv.ProgressBar(dataset_length)
        if (not os.path.exists(self.seg_file)):
            gt_annos = []
            for sample_id in range(dataset_length):
                index_now = self.indices[sample_id][-1]
                rec = self.samples[index_now]
                tran_pose, _, _ = self.get_ego_pose(rec)
                img, rot, tran, intrix, lidar2img = self.get_imgs_full(rec)
                semantic_masks, semantic_indices, semantic_iou, gt_pv_semantic_mask = self.get_semantic_map(rec, lidar2img)
                gt_semantic_map = semantic_indices.numpy()
                gt_anno = {}

                gt_anno['gt_seg'] = gt_semantic_map

                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'seg_GTs': gt_annos
            }
            print('\n GT anns writes to', self.seg_file)
            mmcv.dump(nusc_submissions, self.seg_file)
        else:
            print(f'{self.seg_file} exist, not update')

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def show(self,x,gt=False):
        if gt:
            x1 = (x == 1)
            x2 = (x == 2)
            x3 = (x == 3)
        else:
            x1 = (x==0)
            x2 = (x==1)
            x3 = (x == 2)

        plt.figure()
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])  # 去 y 轴刻度
        plt.imshow(x1, vmin=0, cmap='Blues', vmax=1, alpha=0.6)
        plt.imshow(x2, vmin=0, cmap='Reds', vmax=1, alpha=0.6)
        plt.imshow(x3, vmin=0, cmap='Greens', vmax=1, alpha=0.6)
        plt.show()


def exchange(a, b):
    fft_1 = np.fft.fftshift(np.fft.fftn(a * 1.0))
    fft_2 = np.fft.fftshift(np.fft.fftn(b * 1.0))  ## 中心化

    abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)
    abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)

    fft_1 = abs_2 * np.e ** (1j * angle_1)
    fft_2 = abs_1 * np.e ** (1j * angle_2)
    x1 = np.abs(np.fft.ifftn(fft_1))
    x2 = np.abs(np.fft.ifftn(fft_2))
    return np.int16(x1 / x1.max() * 255), np.int16(x2 / x2.max() * 255)
if __name__ == '__main__':
    data_conf = {
            'xbound': [-30.0, 30.0, 0.3],#[-38.4, 38.4, 0.15],#
            'ybound': [-15.0, 15.0, 0.3],#[-19.2, 19.2, 0.15],#
            'image_size': [512, 512],
            'img':[128,352],
            'thickness': 5,
            'angle_class': 36,
            'w':100,
            'h':200,#0,#
        }
    dataset = NuScenesDataset(version='v1.0-mini', data_conf=data_conf, dataroot='/data0/lsy/dataset/nuScenes/v1.0',is_train=True,test_mode=False,pre=True)#_loss
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    for i in range(0,1):
        res = dataset.__getitem__(i)
        ipm_img = res['img']#3 h w 3
        point = res['point']#3 4 xx
        pre_ipm=res['pre_ipm']
        iou = res['iou']
        imgs = res['cam_img']
        img = [(1 * ipm_img[i]).numpy() for i in range(3)]
        a = np.hstack(img[:3])
        plt.figure()
        plt.imshow(a)
        plt.show()
        # img =[(1*imgs[i]).numpy().transpose(1, 2, 0) for i in range(6)]#6 900 1600 3
        # a = np.hstack(img[:3])
        # b = np.hstack(img[3:])
        # c = np.vstack((a, b))
        # plt.figure()
        # plt.imshow(c)
        # plt.show()
        # #iou[iou>0]=1
        if True:
            plt.figure()
            plt.subplot(131)
            plt.imshow(img[0])
            plt.imshow(iou[1], vmin=0, cmap='Blues', vmax=1, alpha=0.2)
            plt.imshow(iou[2], vmin=0, cmap='Reds', vmax=1, alpha=0.2)
            plt.imshow(iou[3], vmin=0, cmap='Greens', vmax=1, alpha=0.2)
            plt.subplot(132)
            plt.imshow(img[1])
            plt.imshow(iou[1], vmin=0, cmap='Blues', vmax=1, alpha=0.2)
            plt.imshow(iou[2], vmin=0, cmap='Reds', vmax=1, alpha=0.2)
            plt.imshow(iou[3], vmin=0, cmap='Greens', vmax=1, alpha=0.2)
            plt.subplot(133)
            plt.imshow(img[2])
            plt.imshow(iou[1], vmin=0, cmap='Blues', vmax=1, alpha=0.2)
            plt.imshow(iou[2], vmin=0, cmap='Reds', vmax=1, alpha=0.2)
            plt.imshow(iou[3], vmin=0, cmap='Greens', vmax=1, alpha=0.2)
            plt.show()
        fig = plt.figure()
        fig.suptitle('3d')
        ax = fig.gca(projection='3d')
        point = rearrange(point,'n l p -> n p l')
        ipm = rearrange(ipm_img,'n h w c-> n (h w) c')
        ipm = (ipm).numpy()
        num = len(point[0])
        for i in range(len(point)):
            for j in range(0,num,15):#range(2000,3000):#
                if ipm[i,j,0]<1 and ipm[i,j,1]<1 and ipm[i,j,2]<1:
                    ax.scatter(point[i, j,0], point[i,j, 1], point[i,j, 2], color=(ipm[i,j,0],ipm[i,j,1],ipm[i,j,2]), s=5)
        plt.show()

