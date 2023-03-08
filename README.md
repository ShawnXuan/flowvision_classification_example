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
