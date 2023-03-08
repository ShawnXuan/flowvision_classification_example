import os
import cv2
import pickle
import argparse
import numpy as np

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
        default="output/snapshot_epoch0_acc0.6678260869565218",
        help=f"path to snapshot",
    )
    parser.add_argument(
        "--num_classes", type=int, default=23, help="number of classes",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="/data/xiexuan/imagenet/classes_23/val/n09193705/ILSVRC2012_val_00000002.JPEG",
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
    src = cv2.imread(filepath) 
    image = cv2.resize(src, (224, 224))
    image = np.float32(image) / 255.0
    image[:,:,] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    image[:,:,] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    image = image.transpose((2, 0, 1))
    x = flow.from_numpy(image).unsqueeze(0)
    return x


if __name__ == "__main__":
    args = get_args()
    print(args)

    # 加载预训练模型
    assert args.model in model_dict
    model = model_dict[args.model](pretrained=False)

    # 设置类别数, 注意：最后一层必须是`fc`
    assert args.num_classes > 0
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    
    # 导入模型
    assert args.snapshot
    print(f"Loading model from {args.snapshot}")
    state_dict = flow.load(args.snapshot)
    model.load_state_dict(state_dict, strict=True)
    model.to("cuda")

    # 加载训练数据
    x = read_and_transform(args.filepath)
    pred = model(x.cuda())
    pred_index = flow.argmax(pred, 1).numpy()[0]
    print(pred_index)
    with open(args.classes_file, "rb") as f:
        classes = pickle.load(f)
        print(classes)
        print(classes[pred_index])

