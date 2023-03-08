import flowvision
import flowvision.transforms as transforms


model_dict = {
    "resnet50": flowvision.models.resnet50,
    "resnet101": flowvision.models.resnet101,
    "vgg16": flowvision.models.vgg16,
}


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

