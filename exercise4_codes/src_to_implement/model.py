import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """
    A single Residual Block with:
      - (Conv -> BatchNorm -> ReLU) x2
      - Optional 1x1 Conv + BatchNorm for the skip path if input shape changes
      - Summation with skip path, followed by ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1st Conv (with stride)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2nd Conv (no stride)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If stride != 1 or number of channels differ, adjust the skip connection
        self.skip_connection = None
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip path
        if self.skip_connection is not None:
            identity = self.skip_connection(x)

        # Sum and final ReLU
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet variant as specified:

    1) Conv2D(3, 64, 7, 2)
    2) BatchNorm()
    3) ReLU()
    4) MaxPool(3, 2)
    5) ResBlock(64 -> 64, stride=1)
    6) ResBlock(64 -> 128, stride=2)
    7) ResBlock(128 -> 256, stride=2)
    8) ResBlock(256 -> 512, stride=2)
    9) GlobalAvgPool()
    10) Flatten()
    11) FC(512 -> 2)
    12) Sigmoid()
    """

    def __init__(self):
        super().__init__()

        # Initial convolution: Conv2D(3, 64, kernel_size=7, stride=2)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool(3,2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResBlocks
        self.layer1 = ResBlock(64, 64, stride=1)
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)

        # Global Average Pool (to 1×1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully-connected layer: FC(512 -> 2)
        self.fc = nn.Linear(512, 2)

        # Sigmoid output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pool, then flatten
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # FC -> Sigmoid
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# Quick check (optional)
'''if __name__ == "__main__":
    model = ResNet()
    # Suppose you have 300x300 grayscale images turned into 3 channels
    sample_input = torch.randn(8, 3, 300, 300)  # Batch of 8
    output = model(sample_input)
    print("Output shape:", output.shape)  # Expect [8, 2]'''
