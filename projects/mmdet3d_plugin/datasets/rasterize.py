import cv2
import numpy as np

from shapely import affinity
from shapely.geometry import LineString, box,LinearRing,MultiLineString
import torch

def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box#0 0 60 30

    x_min = patch_x - patch_w / 2.0#-30
    y_min = patch_y - patch_h / 2.0#-15
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(
        patch_x, patch_y), use_radians=False)

    return patch


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)

    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(mask, [coords[i:]], False, color=get_discrete_degree(
                coords[i + 1] - coords[i], angle_class=angle_class), thickness=thickness)
    return mask, idx


def line_geom_to_mask(layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type='index', angle_class=36):
    patch_x, patch_y, patch_h, patch_w = local_box#0 0 60 30

    patch = get_patch_coord(local_box)#(-30,-15,30,15)

    canvas_h = canvas_size[0]#200
    canvas_w = canvas_size[1]#100
    scale_height = canvas_h / patch_h#0.3
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0#-0+15
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(
                        new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(
                    new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C-1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def preprocess_map(vectors, patch_size, canvas_size, max_channel, thickness, angle_class):

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])  # 0 0 60 30
    patch_x, patch_y, patch_h, patch_w = local_box  # 0 0 60 30
    canvas_h = canvas_size[0]  # 200
    canvas_w = canvas_size[1]  # 100
    scale_height = canvas_h / patch_h  # 0.3
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0  # -0+15
    trans_y = -patch_y + patch_h / 2.0
    semantic_map=np.zeros(canvas_size, np.uint8)
    #print(len(vectors))
    map = [torch.tensor(np.zeros(canvas_size, np.uint8))]
    for vector in vectors:
        if True:#vector['pts_num'] >= 2:
            map_mask = np.zeros(canvas_size, np.uint8)
            new_line = LineString(vector['pts'][:vector['pts_num']])
            new_line = affinity.affine_transform(
                new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(
                new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            coords = np.asarray(list(new_line .coords), np.int32)
            coords = coords.reshape((-1, 2))
            cv2.polylines(map_mask, [coords], False, color=1, thickness=3)
            cv2.polylines(semantic_map, [coords], False, color= vector['type']+1, thickness=3)
            map.append(torch.tensor(map_mask))
    return torch.stack(map),torch.tensor(semantic_map, dtype=torch.long)


def rasterize_map(vectors, patch_size, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel + 1):
        vector_num_list[i] = []

    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(
                (LineString(vector['pts'][:vector['pts_num']]), vector.get('confidence_level', 1)))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        masks.append(map_mask)

    return np.stack(masks), confidence_levels
