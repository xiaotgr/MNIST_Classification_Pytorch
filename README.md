# MNIST 数据集分类问题

经过 100+ 轮的训练，准确率达到 99.07%

<img src="https://github.com/xiaoaug/MNIST_Classification_Pytorch/assets/39291338/7e88ddc8-58e0-4399-99fe-f9f7d48b8547" width="700">

<img src="https://github.com/xiaoaug/MNIST_Classification_Pytorch/assets/39291338/d991c412-6f1d-40a9-ab0c-73c6c5612938" width="350">

目前还不太清楚为何训练中会出现 Accuracy 突降、Loss 突增的问题。

本项目测试环境为 Ubuntu20.04，python 版本为 3.10.13。

# 如何安装？

```
git clone https://github.com/xiaoaug/MNIST_Classification_Pytorch.git  # 下载
cd MNIST_Classification_Pytorch
pip install -r requirements.txt  # 安装
```

# 如何训练？

1. 根据自己的需要修改 train.py 文件中第 11~23 行的参数（默认也可以）。
2. 运行 train.py 即可：`python train.py`。

> 该项目每轮训练中，只要训练准确率比之前高，就会生成 pth 文件。若该轮训练的准确率比以往训练的准确率低，则不生成 pth 文件。

> 运行 train.py 程序会自动下载 MNIST 数据集到项目文件夹内。

# 如何预测？

1. 根据自己的需要修改 predict.py 文件中第 10~16 行的参数（默认也可以）。
3. 运行 predict.py 即可：`python predict.py`。
