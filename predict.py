import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import setting
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    '0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
    '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine'
]
PRED_IMAGE_NAME = '7.png'    # 用于预测的图像文件名
PRED_IMAGE = f"{setting.PRED_DIR}/{PRED_IMAGE_NAME}"  # pred_set 文件夹内的图片，用于预测图像

print('----> Creating Model')
my_model = Model().to(DEVICE)
print('----> Done')
print("----> Loading Checkpoint")
my_model.load_state_dict(torch.load(setting.PTH_FILE, map_location=DEVICE))
print("----> Done")


def get_image():
    """
    读取用于预测的图片，并进行相应的处理
    :return: 处理完成的图片
    """
    print("----> Loading Image")
    image = torchvision.io.read_image(PRED_IMAGE).type(torch.float32)  # 读取图片
    gray = transforms.Grayscale(num_output_channels=1)
    image = gray(image)
    print("----> Image Shape: ", image.shape)
    image = image / 255.0  # 将图像像素值除以 255 使其数据在 [0, 1] 之间
    print("----> Image Resize To: ", (28, 28))
    resize = transforms.Resize(size=(28, 28), antialias=True)  # 调整图片尺寸
    image = resize(image)
    print("----> Image Shape: ", image.shape)
    print("----> Done")

    return image


def predict() -> None:
    """
    预测
    :return: None
    """
    my_model.eval()
    with torch.inference_mode():
        image = get_image()  # 获取图片
        print("----> Start Predicting")
        pred_label = my_model(image.unsqueeze(dim=0).to(DEVICE))
    # print("----> Predict Label: ", pred_label)

    pred_label_softmax = torch.softmax(pred_label, dim=1)     # 转换成概率
    pred_label_idx = torch.argmax(pred_label_softmax, dim=1)  # 获取概率最大的下标
    pred_label_class = CLASS_NAMES[pred_label_idx.cpu()]  # 找出是狗还是猫
    print(f"----> Predict: === {pred_label_class} ===")
    print("----> Done")

    # 打印图像
    print("----> Drawing Image")
    plt.imshow(image.permute(1, 2, 0))  # 需要把图像从 CHW 转成 HWC
    plt.title(f"Predict: {pred_label_class}")
    plt.axis(False)
    plt.show()
    print("----> Done")


if __name__ == '__main__':
    predict()
