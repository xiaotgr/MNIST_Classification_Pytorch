# MNIST 数据集分类问题

经过 100+ 轮的训练，准确率达到 99.07%

<img src="https://github.com/xiaoaug/MNIST_Classification_Pytorch/assets/39291338/7e88ddc8-58e0-4399-99fe-f9f7d48b8547" width="700">

<img src="https://github.com/xiaoaug/MNIST_Classification_Pytorch/assets/39291338/d991c412-6f1d-40a9-ab0c-73c6c5612938" width="350">

目前还不太清楚为何训练中会出现 Accuracy 突降、Loss 突增的问题。

# 如何训练？

1. 在 setting.py 中根据你自己的情况修改参数。
2. 运行 train.py 即可。

# 如何预测？

1. 在 setting.py 中根据你自己的情况修改参数。
2. 在 predict.py 中将第 15 行的 PRED_IMAGE_NAME 进行修改，换成你需要预测的图片名称。
3. 运行 predict.py 即可。
