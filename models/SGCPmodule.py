import torch.nn as nn
import torch
import torch.nn

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MaxPool2d(nn.Module):
    def __init__(self, k=3, s=2, p=1):
        super().__init__()
        self.pool = nn.MaxPool2d(k, s, p)

    def forward(self, x):
        x = self.pool(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=bias, groups=groups)

    def forward(self, x):
        x = self.conv(x)
        return x

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super(SeparableConv, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2d(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class TD_BU_paths(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False):
        super(TD_BU_paths, self).__init__()
        self.epsilon = epsilon
        self.swish = MemoryEfficientSwish()
        self.first_time = first_time
        self.relu = nn.ReLU()

        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2d(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2d(3, 2, 1)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2d(3, 2, 1)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2d(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2d(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        self.conv6_up = SeparableConv(num_channels)
        self.conv5_up = SeparableConv(num_channels)
        self.conv4_up = SeparableConv(num_channels)
        self.conv3_up = SeparableConv(num_channels)
        self.conv4_down = SeparableConv(num_channels)
        self.conv5_down = SeparableConv(num_channels)
        self.conv6_down = SeparableConv(num_channels)
        self.conv7_down = SeparableConv(num_channels)

        self.p4_downsample = MaxPool2d(3, 2)
        self.p5_downsample = MaxPool2d(3, 2)
        self.p6_downsample = MaxPool2d(3, 2)
        self.p7_downsample = MaxPool2d(3, 2)

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


    def forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p6_w1 = self.relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_SW_sum_1 = self.swish(weight[0] * p6_in + weight[1] *nn.Upsample(size=p6_in.size()[2:], mode='nearest')(p7_in))
        p6_up = self.conv6_up(p6_SW_sum_1)

        p5_w1 = self.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_SW_sum_1 = self.swish(weight[0] * p5_in + weight[1] *nn.Upsample(size=p5_in.size()[2:], mode='nearest')(p6_up))
        p5_up = self.conv5_up(p5_SW_sum_1)

        p4_w1 = self.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_SW_sum_1 = self.swish(weight[0] * p4_in + weight[1] *nn.Upsample(size=p4_in.size()[2:], mode='nearest')(p5_up))
        p4_up = self.conv4_up(p4_SW_sum_1)

        p3_w1 = self.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_SW_sum_1 = self.swish(weight[0] * p3_in + weight[1] *nn.Upsample(size=p3_in.size()[2:], mode='nearest')(p4_up))
        p3_out = self.conv3_up(p3_SW_sum_1)

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        p4_w2 = self.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_SW_sum_2 = self.swish(weight[0] * p4_in + weight[1] * nn.Upsample(size=p4_in.size()[2:])(p4_up) + weight[2] * nn.Upsample(size=p4_in.size()[2:])(self.p4_downsample(p3_out)))
        p4_out = self.conv4_down(p4_SW_sum_2)

        p5_w2 = self.relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_SW_sum_2 = self.swish(weight[0] * p5_in + weight[1] * nn.Upsample(size=p5_in.size()[2:])(p5_up) + weight[2] * nn.Upsample(size=p5_in.size()[2:])(self.p5_downsample(p4_out)))
        p5_out = self.conv5_down(p5_SW_sum_2)

        p6_w2 = self.relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_SW_sum_2 = self.swish(weight[0] * p6_in + weight[1] * nn.Upsample(size=p6_in.size()[2:])(p6_up) + weight[2] * nn.Upsample(size=p6_in.size()[2:])(self.p6_downsample(p5_out)))
        p6_out = self.conv6_down(p6_SW_sum_2)

        p7_w2 = self.relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_SW_sum_2 = self.swish(weight[0] * p7_in + weight[1] * nn.Upsample(size=p7_in.size()[2:])(self.p7_downsample(p6_out)))
        p7_out = self.conv7_down(p7_SW_sum_2)
        return p3_out, p4_out, p5_out, p6_out, p7_out


class TD_BU(nn.Module):
    def __init__(self):
        super(TD_BU, self).__init__()
        self.unit = nn.Sequential(TD_BU_paths(64, [40, 112, 320], True))

    def forward(self, x):
        x = self.unit(x)
        return x

def td_bu():
    model = TD_BU()
    return model


class TD(nn.Module):
    def __init__(self, num_channels, epsilon=1e-4):
        super(TD, self).__init__()
        self.epsilon = epsilon
        self.swish = MemoryEfficientSwish()
        self.relu = nn.ReLU()

        self.conv6_up = SeparableConv(num_channels)
        self.conv5_up = SeparableConv(num_channels)
        self.conv4_up = SeparableConv(num_channels)
        self.conv3_up = SeparableConv(num_channels)

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p6_w1 = self.relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_SW_sum_1 = self.swish(weight[0] * p6_in + weight[1] *nn.Upsample(size=p6_in.size()[2:], mode='nearest')(p7_in))
        p6_up = self.conv6_up(p6_SW_sum_1)

        p5_w1 = self.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_SW_sum_1 = self.swish(weight[0] * p5_in + weight[1] *nn.Upsample(size=p5_in.size()[2:], mode='nearest')(p6_up))
        p5_up = self.conv5_up(p5_SW_sum_1)

        p4_w1 = self.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_SW_sum_1 = self.swish(weight[0] * p4_in + weight[1] *nn.Upsample(size=p4_in.size()[2:], mode='nearest')(p5_up))
        p4_up = self.conv4_up(p4_SW_sum_1)

        p3_w1 = self.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_SW_sum_1 = self.swish(weight[0] * p3_in + weight[1] *nn.Upsample(size=p3_in.size()[2:], mode='nearest')(p4_up))
        p3_out = self.conv3_up(p3_SW_sum_1)

        return p3_out, p4_up, p5_up, p6_up, p7_in

def td():
    model = TD(64)
    return model

class SGCPmodule(nn.Module):
    def __init__(self):
        super(SGCPmodule, self).__init__()
        self.td_bu = td_bu()
        self.td = td()
    def forward(self, x):
        #x = [x3, x4, x5]
        x0 = self.td_bu(x)
        x1 = self.td(x0)
        return x1[0]

if __name__ == "__main__":
    net = td_bu()


