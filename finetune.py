import os
import time
import math
import pickle
import argparse
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import flowvision.datasets as datasets
from oneflow.utils.data import DataLoader

from oneflow import optim as optim
from oneflow.optim.lr_scheduler import LambdaLR
from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy

from sklearn.metrics import accuracy_score, confusion_matrix

from utils import model_dict, val_transforms, train_transforms


def get_args():
    parser = argparse.ArgumentParser(
        description="OneFlow flowvision classification demo",
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
        "--num_epochs", type=int, default=50, help="number of finetune epochs",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="number of finetune epochs",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/path/to/dataset",
        help="Dataset root path, must contain subfolder `train` and `val`",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="how many subprocesses to use for data loading. 0 means using the main process",
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Directory to save classes.pkl.",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="log print interval",
    )
    parser.add_argument("--test_io", action="store_true", help="test io speed")
    parser.add_argument(
        "--save_snapshot", action="store_true", help="save checkpoint after evaluation"
    )

    args = parser.parse_args()
    return args


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def r0_print(*contents):
    if flow.env.get_rank() == 0:
        print(*contents)


if __name__ == "__main__":
    args = get_args()
    r0_print(args)

    # 加载训练数据
    batch_size_per_device = args.batch_size // flow.env.get_world_size()
    assert os.path.isdir(args.data_dir), "Dataset folder is not available."
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transforms
    )
    train_sampler = flow.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    num_batches = len(train_loader)
    classes = train_dataset.classes
    if args.test_io:
        r0_print("start test io...")
        num_samples, last_time = 0, time.time()
        for step, (images, labels) in enumerate(train_loader):
            num_samples += labels.shape[0]
            assert args.log_interval
            if (step + 1) % args.log_interval == 0:
                throughput = num_samples / (time.time() - last_time)
                r0_print(f"step {step+1} of {num_batches}, throughput {throughput:.1f} samples/sec")
                num_samples, last_time = 0, time.time()
        exit()

    # 加载验证数据
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_transforms)
    assert classes == val_dataset.classes
    num_classes = len(classes)

    # 保存类别
    if flow.env.get_rank() == 0:
        classes_file = os.path.join(args.output, "classes.pkl")
        with open(classes_file, "wb") as f:
            pickle.dump(classes, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saving classes to {classes_file}")

    val_sampler = flow.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_per_device, sampler=val_sampler)
    val_prefetched = [batch for batch in val_loader]

    if flow.env.get_rank() == 0:
        filepath = os.path.join(args.output, "val_imagepath_label.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(val_dataset.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saving validation image file path and label to {filepath}")

    def to_global(module_or_tensor, sbp=flow.sbp.split(0), device="cuda"):
        return module_or_tensor.to_global(flow.env.all_device_placement(device), sbp)

    def eval(model, val_loader, log_interval=0):
        n_steps = len(val_loader)
        model.eval()
        pred_list, label_list = [], []

        with flow.no_grad():
            r0_print("start evaluation...")
            for step, (images, labels) in enumerate(val_loader):
                # 将图像传递给模型进行评估
                outputs = model(to_global(images))  # (batch_size, n_classes)
                pred = flow.argmax(outputs, dim=-1)  # (batch_size,)
                pred_list.append(pred)
                label_list.append(to_global(labels, device="cpu"))
                if log_interval > 0:
                    if (step + 1) % log_interval == 0:
                        r0_print(f"{step+1} of {n_steps} batches")

        preds = np.concatenate([pred.to("cpu").numpy() for pred in pred_list])
        labels = np.concatenate([label.numpy() for label in label_list])

        # 计算精度
        metric = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        return {"accuarcy": metric, "confusion_matrix": cm, "labels": labels, "preds": preds}

    def train_one_epoch(
        model, train_loader, criterion, optimizer, lr_scheduler, epoch, log_interval
    ):
        n_steps = len(train_loader)
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            outputs = model(to_global(images))  # (batch_size, n_classes)
            loss = criterion(outputs, to_global(labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if log_interval > 0:
                if (step + 1) % log_interval == 0:
                    loss_syn = loss.numpy()
                    r0_print(f"epoch {epoch}, step {step+1} of {n_steps}, loss {loss_syn:.4f}")

    # 加载预训练模型
    assert args.model in model_dict
    model = model_dict[args.model](pretrained=True)

    # 设置类别数, 注意：最后一层必须是`fc`
    assert num_classes > 0
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = to_global(model, sbp=flow.sbp.broadcast)

    def save_model(subdir):
        if not args.output:
            return
        save_path = os.path.join(args.output, subdir)
        r0_print(f"Saving model to {save_path}")
        state_dict = model.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001
    )

    # 学习率调整
    num_warmup_steps = num_batches * args.warmup_epochs
    num_training_steps = num_batches * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to("cuda")
    to_global(criterion, sbp=flow.sbp.broadcast)

    for epoch in range(args.num_epochs):
        train_one_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.log_interval
        )
        metrics = eval(model, val_prefetched)
        r0_print(f"epoch {epoch}", metrics["accuarcy"])
        if args.save_snapshot:
            subdir = f"snapshot_epoch{epoch}_acc{metrics['accuarcy']}"
            save_model(subdir)
            metric_file = f"metric_epoch{epoch}_acc{metrics['accuarcy']}.pkl"
            with open(metric_file, "wb") as f:
                pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
