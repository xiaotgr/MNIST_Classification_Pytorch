import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

import setting


def get_train_data_loader() -> torch.utils.data.DataLoader:
    """
    获取训练数据
    :return: 训练数据
    """
    print('----> Loading Train Data')
    train_transform = transforms.Compose([
        transforms.ToTensor()  # 0~255 像素值转为 0.0~1.0 的 tensor
    ])
    train_datasets = torchvision.datasets.MNIST(
        root=f'{setting.TRAIN_DIR}',  # 训练集目录地址
        train=True,                   # 是否是训练集
        transform=train_transform,    # 图片数据的转换
        download=True                 # 对标签执行变换
    )

    class_names = train_datasets.classes
    class_dict = train_datasets.class_to_idx
    print('----> Train Data Length: ', len(train_datasets))
    print('----> Train Data Class: ', class_names, class_dict)

    # 将训练数据集转换为 DataLoader
    train_dataloader = DataLoader(
        dataset=train_datasets,
        batch_size=setting.BATCH_SIZE,  # 每次训练的样本数
        shuffle=True,  # 打乱训练集的数据
        num_workers=setting.NUM_WORKERS  # 子进程数
    )
    print('----> Done')
    return train_dataloader


def get_test_data_loader() -> torch.utils.data.DataLoader:
    """
    获取测试数据
    :return: 测试数据
    """
    print('----> Loading Test Data')
    test_transform = transforms.Compose([
        transforms.ToTensor()  # 0~255 像素值转为 0.0~1.0 的 tensor
    ])
    test_datasets = torchvision.datasets.MNIST(
        root=f'{setting.TEST_DIR}',   # 训练集目录地址
        train=False,                  # 是否是训练集
        transform=test_transform,     # 图片数据的转换
        download=True                 # 对标签执行变换
    )

    class_names = test_datasets.classes
    class_dict = test_datasets.class_to_idx
    print('----> Test Data Length: ', len(test_datasets))
    print('----> Test Data Class: ', class_names, class_dict)

    # 将测试数据集转换为 DataLoader
    test_dataloader = DataLoader(
        dataset=test_datasets,
        batch_size=setting.BATCH_SIZE,  # 每次训练的样本数
        shuffle=False,  # 测试集不打乱顺序
        num_workers=setting.NUM_WORKERS  # 子进程数
    )
    print('----> Done')
    return test_dataloader


if __name__ == '__main__':
    train_data_loader = get_train_data_loader()
    test_data_loader = get_test_data_loader()
    print('----> Data Loader Length: ', len(train_data_loader), len(test_data_loader))
