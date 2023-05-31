from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.dense = nn.Sequential(
            nn.Linear(in_features=256, out_features=120, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    model = Model()
    print(model)
