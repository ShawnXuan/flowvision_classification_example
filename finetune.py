import os
import time
import argparse
import numpy as np

import oneflow as flow
import oneflow.nn as nn
import flowvision
import flowvision.transforms as transforms
import flowvision.datasets as datasets
from oneflow.utils.data import DataLoader

from oneflow import optim as optim
from oneflow.optim.lr_scheduler import LambdaLR
from flowvision.loss.cross_entropy import LabelSmoothingCrossEntropy

from sklearn.metrics import accuracy_score


model_dict = {
    "resnet50": flowvision.models.resnet50,
    "resnet101": flowvision.models.resnet101,
    "vgg16": flowvision.models.vgg16,
}


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
        "--mode", type=str, default="eager_global", help=f"eager, eager_global",
    )
    parser.add_argument(
        "--num_classes", type=int, default=23, help="number of classes",
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
        "--batch_size", type=int, default=64, help="Validation batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="how many subprocesses to use for data loading. 0 means using the main process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to save Evaluation results(results.pkl).",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="log print interval",
    )
    parser.add_argument("--test_io", action="store_true", help="test io speed")

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

    def to_device_fn(mode):
        def to_device(module_or_tensor, sbp=flow.sbp.split(0)):
            if flow.env.get_world_size() == 1:
                return module_or_tensor.to("cuda")
            if mode == "eager":
                return module_or_tensor.to("cuda")
            elif mode == "eager_global":
                return module_or_tensor.to_global(flow.env.all_device_placement("cuda"), sbp)
            elif mode == "graph":
                assert 0, "graph mode is UNIMPLEMENTED"
            else:
                assert 0

        return to_device

    to_device = to_device_fn(args.mode)

    def eval(model, val_loader, log_interval=0):
        n_steps = len(val_loader)
        model.eval()
        pred_list, label_list = [], []

        with flow.no_grad():
            r0_print("start evaluation...")
            for step, (images, labels) in enumerate(val_loader):
                # 将图像传递给模型进行评估
                outputs = model(to_device(images))  # (batch_size, n_classes)
                pred = flow.argmax(outputs, dim=-1)  # (batch_size,)
                pred_list.append(pred)
                label_list.append(labels)
                if log_interval > 0:
                    if (step + 1) % log_interval == 0:
                        r0_print(f"{step+1} of {n_steps} batches")

        preds = np.concatenate([pred.to("cpu").numpy() for pred in pred_list])
        labels = np.concatenate([label.numpy() for label in label_list])

        # 计算精度
        metric = accuracy_score(labels, preds)
        return {"accuarcy": metric}

    def train_one_epoch(
        model, train_loader, criterion, optimizer, lr_scheduler, epoch, log_interval
    ):
        n_steps = len(train_loader)
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            outputs = model(to_device(images))  # (batch_size, n_classes)
            loss = criterion(outputs, to_device(labels))
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
    assert args.num_classes > 0
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    # model = to_device(model, sbp=flow.sbp.broadcast)
    to_device(model, sbp=flow.sbp.broadcast)

    # 数据预处理
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]
    )
    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 加载训练数据
    assert os.path.isdir(args.data_dir), "Dataset folder is not available."
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), transform=train_transforms
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    num_batches = len(train_loader)
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
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    val_prefetched = [batch for batch in val_loader]

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001
    )

    # 学习率调整
    num_warmup_steps = num_batches * args.warmup_epochs
    num_training_steps = num_batches * args.num_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to("cuda")
    to_device(criterion, sbp=flow.sbp.broadcast)

    for epoch in range(args.num_epochs):
        train_one_epoch(
            model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.log_interval
        )
        metrics = eval(model, val_prefetched)
        r0_print(f"epoch {epoch}", metrics)
