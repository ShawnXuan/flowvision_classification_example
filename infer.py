import os
import cv2
import pickle
import argparse
import numpy as np

from PIL import Image
import oneflow as flow
import oneflow.nn as nn
from utils import model_dict, val_transforms


def get_args():
    parser = argparse.ArgumentParser(
        description="OneFlow flowvision inference demo",
        epilog="Example of use",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        help=f"Supported models: {', '.join(model_dict.keys())}",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="output/snapshot_epoch15_acc0.8573913043478261",
        help=f"path to snapshot",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="val/n10565667/ILSVRC2012_val_00000255.JPEG",
        help="path to an image file",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="output/classes.pkl",
        help="path to classes file",
    )

    args = parser.parse_args()
    return args


def read_and_transform(filepath):
    # https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py#L244
    with open(filepath, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
        return val_transforms(img).unsqueeze(0)


if __name__ == "__main__":
    args = get_args()
    print(args)
    
    device = "cuda" if flow.cuda.is_available() else "cpu"

    # 加载类别列表
    with open(args.classes_file, "rb") as f:
        classes = pickle.load(f)
        num_classes = len(classes)

    # 加载预训练模型
    assert args.model in model_dict
    model = model_dict[args.model](pretrained=False)

    # 设置类别数, 注意：最后一层必须是`fc`
    assert num_classes > 0
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 导入模型
    assert args.snapshot
    print(f"Loading model from {args.snapshot}")
    state_dict = flow.load(args.snapshot)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 加载训练数据
    x = read_and_transform(args.filepath)
    pred = model(x.to(device))
    pred_index = flow.argmax(pred, 1).numpy()[0]
    print(f"prediction index:{pred_index}, prediction class: {classes[pred_index]}")

