## 0 安装tabulate
```bash
wget https://pypi.tuna.tsinghua.edu.cn/packages/40/44/4a5f08c96eb108af5cb50b41f76142f0afa346dfa99d5296fe7202a11854/tabulate-0.9.0-py3-none-any.whl
python -m pip install tabulate-0.9.0-py3-none-any.whl
```

## 1 安装flowvision
### 获取flowvision

- 方式1: git clone
```bash
git clone https://github.com/Oneflow-Inc/vision.git
cd vision
```
- 方式2: wget zip 
```bash
wget https://github.com/Oneflow-Inc/vision/archive/refs/heads/main.zip
unzip main.zip
cd vision-main
```

### pip安装命令
```bash
python -m pip install -e .
```

## 2 下载resnet50预训练模型
```bash
mkdir -p /root/.oneflow/flowvision_cache
cd /root/.oneflow/flowvision_cache
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/flowvision/classification/ResNet/resnet50.zip
unzip resnet50.zip
```

删除resnet50.zip（可选）

## 3. 精调 resnet50
```bash
./run.sh 1 /path/to/dataset
```

## 参数说明

### finetune.py

| 参数名称        | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| --model         | 支持网络模型名称，可选项来自`util.py`里面的字典`model_dict`，其实可以支持更多 |
| --num_classes   | 分类数，比如标准的imagenet有1000类，这个参数会用于修改模型的最后一层`fc` |
| --num_epochs    | 训练集循环的次数                                             |
| --warmup_epochs | 学习率预热多少周期                                           |
| --data_dir      | 数据集的根目录，注意：改目录下需要包括`train`和`val`两个子目录，这两个子目录里面是按照类别目录分别存放的图片 |
| --batch_size    | 每个训练批次所有设备中图片的数量，即：global_batch_size      |
| --num_workers   | 使用多少子进程加载和预处理数据， 0表示使用主进程。目前训练的瓶颈基本就在数据预处理上 |
| --output        | 指定一个目录用于保存：<br>- 模型的snapshot<br>- classes.pkl，这是保存了数据集分类的名称列表，本例子中使用了`flowvision.datasets.ImageFolder`加载数据目录，这个列表按照顺序保存了分类的名称，如：<br>['n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815'] |
| --log_interval  | 打印日志的间隔                                               |
| --test_io       | 这是一个开关选项，打开时只测试io图片加载的速度               |
| --save_snapshot | 这是一个开关选项，打开时，每个epoch都会保存一个snapshot到`--output`指定的目录 |

### infer.py

| 参数名称       | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| --model        | 支持网络模型名称，可选项来自`util.py`里面的字典`model_dict`，其实可以支持更多 |
| --snapshot     | 待加载的模型地址                                             |
| --num_classes  | 分类数，比如标准的imagenet有1000类，这个参数会用于修改模型的最后一层`fc` |
| --filepath     | 待推断的图片地址                                             |
| --classes_file | 分类列表文件地址                                             |



