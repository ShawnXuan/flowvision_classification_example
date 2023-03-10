import os
import cv2
import pickle
import argparse
import numpy as np

from PIL import Image
import oneflow as flow
import oneflow.nn as nn
from utils import model_dict, val_transforms
from sklearn.metrics import accuracy_score, confusion_matrix

import flowvision.datasets as datasets
from oneflow.utils.data import DataLoader


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
        "--data_dir",
        type=str,
        default="/path/to/val",
        help="path to val folder",
    )
    parser.add_argument(
        "--classes_file",
        type=str,
        default="output/classes.pkl",
        help="path to classes file",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="batch size",
    )

    args = parser.parse_args()
    return args


def eval(model, val_loader, log_interval=0):
    n_steps = len(val_loader)
    model.eval()
    pred_list, label_list = [], []

    with flow.no_grad():
        print("start evaluation...")
        for step, (images, labels) in enumerate(val_loader):
            # 将图像传递给模型进行评估
            outputs = model(images.cuda())  # (batch_size, n_classes)
            pred = flow.argmax(outputs, dim=-1)  # (batch_size,)
            pred_list.append(pred)
            label_list.append(labels)
            if log_interval > 0:
                if (step + 1) % log_interval == 0:
                    print(f"{step+1} of {n_steps} batches")

    preds = np.concatenate([pred.to("cpu").numpy() for pred in pred_list])
    labels = np.concatenate([label.numpy() for label in label_list])

    # 计算精度
    metric = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return {"accuarcy": metric, "confusion_matrix": cm, "labels": labels, "preds": preds}


if __name__ == "__main__":
    args = get_args()
    print(args)

    # 加载验证数据
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transforms)
    num_classes = len(val_dataset.classes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    val_prefetched = [batch for batch in val_loader]

    # 加载模型
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
    model.to("cuda")

    metric = eval(model, val_prefetched)
    print(metric)

