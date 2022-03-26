import torch.nn as nn
from models.mbnet import mbnet
from models.SGCPmodule import SGCPmodule

class SGCPNet(nn.Module):
    def __init__(self, num_classes=19):
        super(SGCPNet, self).__init__()

        mb = mbnet()
        mb = list(mb.children())
        mb_blocks = list(mb[3].children())

        self.l1 = nn.Sequential(*mb[:3])
        self.l2 = mb_blocks[0]
        self.l3 = nn.Sequential(*mb_blocks[1:3])
        self.l4 = nn.Sequential(*mb_blocks[3:8])
        self.l5 = nn.Sequential(*mb_blocks[8:])

        self.conv3 = nn.Conv2d(24, 40, 1, 1, 0)
        self.conv4 = nn.Conv2d(48, 112, 1, 1, 0)
        self.conv5 = nn.Conv2d(96, 320, 1, 1, 0)

        self.sgcp = SGCPmodule()
        self.classifier = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, x):

        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        x5 = self.l5(x4)

        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        p3=self.sgcp([x3, x4, x5])

        output = self.classifier(p3)
        return output


def sgcpnet(num_classes=19):
    net = SGCPNet(num_classes)
    return net

if __name__ == "__main__":
    net = SGCPNet()