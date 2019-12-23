"""
transform_compose = transforms.Compose([
                       transforms.Resize((84, 84)), #redimension
                       transforms.ToTensor()
                    ])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=85)

Training complete in 8m 39s
Best accuracy on epoch 93: 0.891179
Accuracy of the network on the test images: 82.02%
"""
import torch.nn as nn
import torch.nn.functional as F

class KutralNet(nn.Module): # test 12
    def __init__(self, classes, initial_filters=32):
        super(KutralNet, self).__init__()
        self.firstBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=initial_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=initial_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        expansion = 4
        n_filters = initial_filters * 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=initial_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        initial_filters = n_filters
        n_filters = initial_filters * 2

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=initial_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        initial_filters = n_filters
        n_filters = initial_filters // 2

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=initial_filters, out_channels=n_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=n_filters)
        )

        self.down_sample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(num_features=n_filters)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(n_filters, classes)

        self._init_params()

    def forward(self, x):
        debug = False
        if debug:
            print('x.size()', x.size(x))
        x = self.firstBlock(x)
        if debug:
            print('firstBlock.size()', x.size())
        shortcut = self.block1(x)
        if debug:
            print('block1.size()', x.size())
        x = self.block2(shortcut)
        if debug:
            print('block2.size()', x.size())
        x = self.block3(x)
        if debug:
            print('block3.size()', x.size())
        x += self.down_sample(shortcut)
        # global average pooling
        x = self.global_pool(F.leaky_relu(x))
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
