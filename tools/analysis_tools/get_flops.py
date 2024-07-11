# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import torch
from mmcv import Config, DictAction
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets import custom_build_dataset
from mmdet3d.models import build_model
import os
from os import path as osp
from mmcv.cnn import add_flops_counting_methods
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',default="/data0/lsy/4.v2_MapTR-main/work_dirs/run_18/v4maptr_tiny_r50_24e.py", help='test config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='image',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args
def get_model_complexity_info(model,
                              input_shape,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              flush=False,
                              ost=sys.stdout):
    """Get complexity information of a model.

    This method can calculate FLOPs and parameter counts of a model with
    corresponding input shape. It can also print complexity information for
    each layer in a model.

    Supported layers are listed as below:
        - Convolutions: ``nn.Conv1d``, ``nn.Conv2d``, ``nn.Conv3d``.
        - Activations: ``nn.ReLU``, ``nn.PReLU``, ``nn.ELU``, ``nn.LeakyReLU``,
            ``nn.ReLU6``.
        - Poolings: ``nn.MaxPool1d``, ``nn.MaxPool2d``, ``nn.MaxPool3d``,
            ``nn.AvgPool1d``, ``nn.AvgPool2d``, ``nn.AvgPool3d``,
            ``nn.AdaptiveMaxPool1d``, ``nn.AdaptiveMaxPool2d``,
            ``nn.AdaptiveMaxPool3d``, ``nn.AdaptiveAvgPool1d``,
            ``nn.AdaptiveAvgPool2d``, ``nn.AdaptiveAvgPool3d``.
        - BatchNorms: ``nn.BatchNorm1d``, ``nn.BatchNorm2d``,
            ``nn.BatchNorm3d``, ``nn.GroupNorm``, ``nn.InstanceNorm1d``,
            ``InstanceNorm2d``, ``InstanceNorm3d``, ``nn.LayerNorm``.
        - Linear: ``nn.Linear``.
        - Deconvolution: ``nn.ConvTranspose2d``.
        - Upsample: ``nn.Upsample``.

    Args:
        model (nn.Module): The model for complexity calculation.
        input_shape (tuple): Input shape used for calculation.
        print_per_layer_stat (bool): Whether to print complexity information
            for each layer in a model. Default: True.
        as_strings (bool): Output FLOPs and params counts in a string form.
            Default: True.
        input_constructor (None | callable): If specified, it takes a callable
            method that generates input. otherwise, it will generate a random
            tensor with input shape to calculate FLOPs. Default: None.
        flush (bool): same as that in :func:`print`. Default: False.
        ost (stream): same as ``file`` param in :func:`print`.
            Default: sys.stdout.

    Returns:
        tuple[float | str]: If ``as_strings`` is set to True, it will return
            FLOPs and parameter counts in a string format. otherwise, it will
            return those in a float number format.
    """
    assert type(input_shape) is tuple
    assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    if input_constructor:
        input = input_constructor(input_shape)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty(
                (1, *input_shape),
                dtype=next(flops_model.parameters()).dtype,
                device=next(flops_model.parameters()).device)
        except StopIteration:
            # Avoid StopIteration for models which have no parameters,
            # like `nn.Relu()`, `nn.AvgPool2d`, etc.
            batch = torch.ones(()).new_empty((1, *input_shape))

        _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model, flops_count, params_count, ost=ost, flush=flush)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count

def main():

    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
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
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    dataset = custom_build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))
    for i, data in enumerate(data_loader):
        if i>0:
            break
        flops_model = add_flops_counting_methods(model)
        flops_model.eval()
        flops_model.start_flops_count()
        _ = flops_model(return_loss=False, rescale=True, **data)
        flops_count, params_count = flops_model.compute_average_flops_cost()
        if True:
            print_model_with_flops(
                flops_model, flops_count, params_count, ost=ost, flush=flush)
        flops_model.stop_flops_count()

        flops, params=flops_to_string(flops_count), params_to_string(params_count)

    split_line = '=' * 30
    print(f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
