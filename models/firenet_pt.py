"""
64x64
Training complete in 5m 17s
Best val Acc: 0.882030
"""
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, features_in, features_out):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(features_in, features_out, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout2d(p=.5)
    # end __init__

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.avg_pool(x)
        return self.dropout(x)
    # end forward
# end ConvBlock

class FireNet(nn.Module):
    def __init__(self, classes):
        super(FireNet, self).__init__()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)
        self.linear1 = nn.Linear(64 * 6 * 6, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=.2)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, classes)
        # self.softmax = nn.Softmax(dim=1)
        # self._init_params()
    # end __init__

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = x.view(x.size()[0], -1) # flatten
        x = x.flatten(start_dim=1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        # x = self.relu(self.linear3(x))
        # x = self.softmax(x)
        return x
    # end forward

    # def _init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform_(m.weight)#, gain=nn.init.calculate_gain('relu'))
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)
    # end _init_params
# end FireNet
