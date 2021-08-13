#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script.
"""
import argparse
import os
import torch
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
)
from utils import TRAR_Preprocess
import numpy as np

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper['coco_2014_train'] = 'train2014'
dataset_to_folder_mapper['coco_2014_val'] = 'val2014'
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper['coco_2015_test'] = 'test2015'

def extract_grid_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2014_train",
                        choices=['coco_2014_train', 'coco_2014_val', 'coco_2015_test'])
    parser.add_argument("--output_dir", type=str, default="", metavar="PATH", required=True, help="path to save the extracted feature")
    parser.add_argument("--weight_path", type=str, default="", metavar="FILE", required=True, help="path to the pretrained model weight")
    parser.add_argument("--feature_size", type=int, default=8, help="downsample ratios")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def extract_grid_feature_on_dataset(model, data_loader, dump_folder, args):
    for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            image_id = inputs[0]['image_id']
            # file_name = '%d.pth' % image_id
            file_name = '%s.npy' % image_id
            # compute features
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            outputs = model.roi_heads.get_conv5_features(features) # (1, 2048, h, w)

            # you can add some operation to control the outputs, like conv2d, maxpool2d, avgpool2d
            # save as np.float16 can help you to save more memory
            
            # TRAR Pre-process
            outputs = TRAR_Preprocess(outputs, feature_size=args.feature_size)
            outputs = outputs.astype(np.float16)

            with PathManager.open(os.path.join(dump_folder, file_name), "wb") as f:
                # save as np.ndarray
                np.save(f,outputs)

def do_feature_extraction(cfg, model, args):
    dataset_name = args.dataset
    with inference_context(model):
        # edit config file
        cfg.defrost()
        cfg.OUTPUT_DIR = args.output_dir
        cfg.freeze()

        dump_folder = os.path.join(cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name])
        PathManager.mkdirs(dump_folder)
        data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)
        extract_grid_feature_on_dataset(model, data_loader, dump_folder, args)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    # edit config file
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.weight_path
    cfg.freeze()

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    do_feature_extraction(cfg, model, args)


if __name__ == "__main__":
    args = extract_grid_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
