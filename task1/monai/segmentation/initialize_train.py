'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
#%%
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    DeleteItemsd,
    Spacingd,
    RandAffined,
    Rand3DElasticd,
    ConcatItemsd,
    ScaleIntensityRanged,
    RandSpatialCropd,
    RandAdjustContrastd,
    RandGaussianSharpend,
    RandGaussianNoised,
    Invertd,
    AsDiscreted,
    SaveImaged,
)
import torch.nn as nn 
from monai.networks.nets import UNet#, SegResNet, DynUNet, SwinUNETR, UNETR, AttentionUnet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, GeneralizedDiceFocalLoss
import torch
import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
import sys 
import torch.nn.functional as F
config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(config_dir)
from config import DATA_FOLDER, WORKING_FOLDER
#%%
def pad_zeros_at_front(num, N):
    return  str(num).zfill(N)

def create_dictionary_ctptgt(ctpaths, ptpaths, gtpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'GT':gtpath})
    return data

def create_dictionary_ctptgtpr(ctpaths, ptpaths, gtpaths, prpaths):
    data = []
    for i in range(len(gtpaths)):
        ctpath = ctpaths[i]
        ptpath = ptpaths[i]
        gtpath = gtpaths[i]
        prpath = prpaths[i]
        data.append({'CT':ctpath, 'PT':ptpath, 'GT':gtpath, 'PR': prpath})
    return data

def remove_all_extensions(filename):
    while True:
        name, ext = os.path.splitext(filename)
        if ext == '':
            return name
        filename = name

#%%
def get_train_valid_data_in_dict_format(fold):
    data_split_fpath = os.path.join(WORKING_FOLDER, 'data_analysis/data_splits.json')
    with open(data_split_fpath, 'r') as file:
        split_data = json.load(file)
    train_ids = split_data[fold]['train']
    valid_ids = split_data[fold]['val']

    ctpaths_train = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0000.nii.gz') for id in train_ids]
    ptpaths_train = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0001.nii.gz') for id in train_ids]
    gtpaths_train = [os.path.join(DATA_FOLDER, 'labelsTr', f'{id}.nii.gz') for id in train_ids]

    ctpaths_valid = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0000.nii.gz') for id in valid_ids]
    ptpaths_valid = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0001.nii.gz') for id in valid_ids]
    gtpaths_valid = [os.path.join(DATA_FOLDER, 'labelsTr', f'{id}.nii.gz') for id in valid_ids]

    train_data = create_dictionary_ctptgt(ctpaths_train, ptpaths_train, gtpaths_train)
    valid_data = create_dictionary_ctptgt(ctpaths_valid, ptpaths_valid, gtpaths_valid)

    return train_data, valid_data


def get_valid_pred_data_in_dict_format(fold, pred_folder):
    data_split_fpath = os.path.join(WORKING_FOLDER, 'data_analysis/data_splits.json')
    with open(data_split_fpath, 'r') as file:
        split_data = json.load(file)
    valid_ids = split_data[fold]['val']

    ctpaths_valid = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0000.nii.gz') for id in valid_ids]
    ptpaths_valid = [os.path.join(DATA_FOLDER, 'imagesTr', f'{id}_0001.nii.gz') for id in valid_ids]
    gtpaths_valid = [os.path.join(DATA_FOLDER, 'labelsTr', f'{id}.nii.gz') for id in valid_ids]
    prpaths_valid = [os.path.join(pred_folder, f'{id}.nii.gz') for id in valid_ids]

    data = create_dictionary_ctptgtpr(ctpaths_valid, ptpaths_valid, gtpaths_valid, prpaths_valid)

    return data
#%%
# def get_test_data_in_dict_format():
#     test_fpaths = os.path.join(WORKING_FOLDER, 'data_split/test_filepaths.csv')
#     test_df = pd.read_csv(test_fpaths)
#     ctpaths_test, ptpaths_test, gtpaths_test = list(test_df['CTPATH'].values), list(test_df['PTPATH'].values),  list(test_df['GTPATH'].values)
#     test_data = create_dictionary_ctptgt(ctpaths_test, ptpaths_test, gtpaths_test)
#     return test_data

def get_spatial_size(input_patch_size):
    return (input_patch_size, input_patch_size, input_patch_size)

def get_spacing():
    spc = 2
    return (spc, spc, spc)

def get_train_transforms(input_patch_size):
    spatialsize = get_spatial_size(input_patch_size)
    spacing = get_spacing()
    mod_keys = ['CT', 'PT', 'GT']
    train_transforms = Compose(
    [
        LoadImaged(keys=mod_keys, image_only=True),
        EnsureChannelFirstd(keys=mod_keys),
        CropForegroundd(keys=mod_keys, source_key='CT'),
        ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'bilinear', 'nearest')),
        RandSpatialCropd(
            keys=mod_keys,
            roi_size=spatialsize,
            random_center=True,
            random_size=False,
        ),
        RandAffined(
            keys=mod_keys,
            mode=('bilinear', 'bilinear', 'nearest'),
            prob=0.5,
            spatial_size = spatialsize,
            translate_range=(10,10,10),
            rotate_range=(0, 0, np.pi/12),
            scale_range=(0.1, 0.1, 0.1)),
        Rand3DElasticd(
            keys=mod_keys,
            sigma_range=(0.0, 1.0),
            magnitude_range=(0.0, 1.0),
            spatial_size = spatialsize,
            prob=0.5,
        ),
        RandAdjustContrastd(
            keys = ['CT', 'PT'],
            prob = 0.3,
            gamma = (0.70, 1.5),
        ),
        RandGaussianNoised(
            keys=['CT', 'PT'],
            prob=0.5,
        ),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        DeleteItemsd(keys=['CT', 'PT'])
    ])

    return train_transforms
#%%
def get_valid_transforms():
    spacing = get_spacing()
    mod_keys = ['CT', 'PT', 'GT']
    valid_transforms = Compose(
    [
        LoadImaged(keys=mod_keys),
        EnsureChannelFirstd(keys=mod_keys),
        ScaleIntensityRanged(keys=['CT'], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=mod_keys, source_key='CT'),
        Orientationd(keys=mod_keys, axcodes="RAS"),
        Spacingd(keys=mod_keys, pixdim=spacing, mode=('bilinear', 'bilinear', 'nearest')),
        ConcatItemsd(keys=['CT', 'PT'], name='CTPT', dim=0),
        DeleteItemsd(keys=['CT', 'PT'])
    ])
    return valid_transforms


def get_post_transforms(test_transforms, save_preds_dir):
    post_transforms = Compose([
        Invertd(
            keys="Pred",
            transform=test_transforms,
            orig_keys="GT",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="Pred", argmax=True),
        SaveImaged(keys="Pred", meta_keys="pred_meta_dict", output_dir=save_preds_dir, output_postfix="", separate_folder=False, resample=False),
    ])
    return post_transforms

def get_kernels_strides(patch_size, spacings):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes, spacings = patch_size, spacings
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides
#%%
def get_model(network_name = 'unet', input_patch_size=192):
    if network_name == 'unet':
        model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )
    # elif network_name == 'swinunetr':
    #     spatialsize = get_spatial_size(input_patch_size)
    #     model = SwinUNETR(
    #         img_size=spatialsize,
    #         in_channels=2,
    #         out_channels=2,
    #         feature_size=12,
    #         use_checkpoint=False,
    #     )
    # elif network_name =='segresnet':
    #     model = SegResNet(
    #         spatial_dims=3,
    #         blocks_down=[1, 2, 2, 4],
    #         blocks_up=[1, 1, 1],
    #         init_filters=16,
    #         in_channels=2,
    #         out_channels=2,
    #     )
    # elif network_name == 'dynunet':
    #     spatialsize = get_spatial_size(input_patch_size)
    #     spacing = get_spacing()
    #     krnls, strds = get_kernels_strides(spatialsize, spacing)
    #     model = DynUNet(
    #         spatial_dims=3,
    #         in_channels=2,
    #         out_channels=2,
    #         kernel_size=krnls,
    #         strides=strds,
    #         upsample_kernel_size=strds[1:],
    #     )
    # else:
    #     pass
    return model

#%%
class WeightedDiceLoss3D(nn.Module):
    def __init__(self, weight_fp=1.0, weight_fn=1.0):
        super(WeightedDiceLoss3D, self).__init__()
        self.weight_fp = weight_fp
        self.weight_fn = weight_fn
    
    def forward(self, inputs, targets, smooth=1):
        # Apply softmax and select the foreground class (assume class 1 is the target class)
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs[:, 1, :, :, :]

        # Flatten the tensors for easier computation
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Calculate false positives and false negatives
        false_positives = (inputs * (1 - targets)).sum()
        false_negatives = ((1 - inputs) * targets).sum()

        # Calculate weighted dice loss
        weighted_dice_loss = - dice + \
            self.weight_fp * (false_positives / (union + smooth)) + \
            self.weight_fn * (false_negatives / (union + smooth))
        # weighted_dice_loss += self.weight_fp * (false_positives / (union + smooth))
        # weighted_dice_loss += self.weight_fn * (false_negatives / (union + smooth))

        return weighted_dice_loss
#%%
def get_gendicefocalloss_function():
    loss_function = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True, weight=[1., 100.], lambda_gdl=1.0, lambda_focal=1.0)
    return loss_function

def get_loss_function():
    # loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    # loss_function = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True, weight=[1., 100.], lambda_gdl=1.0, lambda_focal=1.0)
    loss_function = WeightedDiceLoss3D(weight_fp=100.0, weight_fn=0.0)
    return loss_function

def get_optimizer(model, learning_rate=2e-4, weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def get_metric():
    metric = DiceMetric(include_background=False, reduction="mean")
    return metric

def get_scheduler(optimizer, max_epochs=500):
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)
    return scheduler

def get_validation_sliding_window_size(inference_patch_size):
    window_size = get_spatial_size(inference_patch_size)
    return window_size
# %%
