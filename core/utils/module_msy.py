import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.add_module import SELayer
from core.utils.utils import bilinear_sampler


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_fn='batch', stride=1):
        super(ResBlock, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            # self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm1 = nn.GroupNorm(num_groups=4, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(output_dim)
            self.norm2 = nn.BatchNorm2d(output_dim)
            self.norm_down = nn.BatchNorm2d(output_dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(output_dim)
            self.norm2 = nn.InstanceNorm2d(output_dim)
            self.norm_down = nn.InstanceNorm2d(output_dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride),
                                            self.norm_down)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class FeaExt(nn.Module):
    def __init__(self, output_dim=256, norm_fn='batch', dropout=0.0):
        super(FeaExt, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=96)
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=128)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(96)
            self.norm3 = nn.BatchNorm2d(128)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(96)
            self.norm3 = nn.InstanceNorm2d(128)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), self.norm1, self.relu1,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        )  # (B, 64, H, W)
        self.conv2 = ResBlock(64, 96, norm_fn=self.norm_fn, stride=2)  # (B, 96, H/2, W/2)
        self.conv3 = ResBlock(96, 128, norm_fn=self.norm_fn, stride=2)  # (B, 128, H/4, W/4)
        self.conv4 = ResBlock(128, output_dim, norm_fn=self.norm_fn, stride=2)  # (B, 256, H/8, W/8)
        # self.conv4 = nn.Conv2d(128, output_dim, kernel_size=1) 原始网络的步骤

        # SE_block
        self.se_block = SELayer(output_dim)

        # 处理各层的特征，即将2B融合成1B
        self.layer1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1)
        self.layer2_conv = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.layer3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        layer1 = self.relu1(self.norm1(self.conv1(x)))
        layer2 = self.relu1(self.norm2(self.conv2(layer1)))
        layer3 = self.relu1(self.norm3(self.conv3(layer2)))
        layer4 = self.relu1(self.conv4(layer3))

        x = self.se_block(layer4)
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
            layer1 = torch.split(layer1, [batch_dim, batch_dim], dim=0)
            layer2 = torch.split(layer2, [batch_dim, batch_dim], dim=0)
            layer3 = torch.split(layer3, [batch_dim, batch_dim], dim=0)

            layer1 = torch.cat([layer1[0], layer1[1]], dim=1)
            layer2 = torch.cat([layer2[0], layer2[1]], dim=1)
            layer3 = torch.cat([layer3[0], layer3[1]], dim=1)

            layer1 = self.layer1_conv(layer1)
            layer2 = self.layer2_conv(layer2)
            layer3 = self.layer3_conv(layer3)
            # feature_pyramid1.append(layer4[0]), feature_pyramid2.append(layer4[1])
            # feature_pyramid1.append(layer3[0]), feature_pyramid2.append(layer3[1])
            # feature_pyramid1.append(layer2[0]), feature_pyramid2.append(layer2[1])
            # feature_pyramid1.append(layer1[0]), feature_pyramid2.append(layer1[1])
            return x[0], x[1], layer1, layer2, layer3
        else:
            return x, layer1, layer2, layer3

        # return x, layer1, layer2, layer3, layer4


class FlowEstimator(nn.Module):
    """输入应该为corr、image1的特征图、预测流flow"""

    def __init__(self, input_dim):
        super(FlowEstimator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=1), nn.BatchNorm2d(128), self.relu,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), self.relu)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=1), nn.BatchNorm2d(96), self.relu,
            nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(96), self.relu)
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1), nn.BatchNorm2d(64), self.relu,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), self.relu)
        self.predict_flow = nn.Conv2d(64, 2, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        """x3作为Context Network的信息"""
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        flow = self.predict_flow(x3)
        return x3, flow


class ContextNetwork(nn.Module):
    """
    可以作为两个用途，为update_block模块准备输入（来自RAFT代码的逻辑），和为Flow refine准备输入（来自ARFlow的逻辑）,
    但可能由于FeaExt已经提取了fmap1或2，所以此处为Flow refine做准备
    """

    def __init__(self, input_dim):
        super(ContextNetwork, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, 1, 1), nn.BatchNorm2d(128), self.relu,
            nn.Conv2d(128, 96, 3, 1, 1), nn.BatchNorm2d(96), self.relu,
            nn.Conv2d(96, 64, 3, 1, 1), nn.BatchNorm2d(64), self.relu,
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), self.relu,
            nn.Conv2d(32, 2, 3, 1, 1)
        )

    def forward(self, x):
        return x


class DownFeatureChannels(nn.Module):
    def __init__(self, num_layer=4):
        super(DownFeatureChannels, self).__init__()
        self.num_layer = num_layer
        self.conv_1_1 = nn.ModuleList(
            [nn.Conv2d(256, 32, kernel_size=1),
             nn.Conv2d(128, 32, kernel_size=1),
             nn.Conv2d(96, 32, kernel_size=1),
             nn.Conv2d(64, 32, kernel_size=1)])
        self.norm = nn.ModuleList([
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),
        ])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fpyramid):
        feature_list = []
        for i in range(self.num_layer):
            feature = self.relu(self.norm[i](self.conv_1_1[i](fpyramid[i])))
            feature_list.append(feature)
        return feature_list


class UpscaleFeature(nn.Module):
    """将输入的层，先利用1*1卷积减少通道数，再上采样到全分辨率，然后拼接起来"""

    def __init__(self, num_layer=4):
        super(UpscaleFeature, self).__init__()
        self.num_layer = num_layer
        self.deconv = nn.ModuleList([
            nn.ConvTranspose2d(32, 32, kernel_size=8, padding=0, stride=8, output_padding=0),
            nn.ConvTranspose2d(32, 32, kernel_size=4, padding=0, stride=4, output_padding=0),
            nn.ConvTranspose2d(32, 32, kernel_size=2, padding=0, stride=2, output_padding=0),
            nn.ConvTranspose2d(32, 32, kernel_size=1, padding=0, stride=1, output_padding=0),
        ])
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1)
        # self.conv_1_1 = nn.ModuleList(
        #     [nn.Conv2d(256, 32, kernel_size=1),
        #      nn.Conv2d(128, 32, kernel_size=1),
        #      nn.Conv2d(96, 32, kernel_size=1),
        #      nn.Conv2d(64, 32, kernel_size=1)])
        # self.norm = nn.ModuleList([
        #     nn.BatchNorm2d(32),
        #     nn.BatchNorm2d(32),
        #     nn.BatchNorm2d(32),
        #     nn.BatchNorm2d(32),
        # ])

    def forward(self, fpyramid0, fpyramid1, fpyramid2, fpyramid3):
        fpyramid0 = self.deconv[0](fpyramid0)
        fpyramid1 = self.deconv[1](fpyramid1)
        fpyramid2 = self.deconv[2](fpyramid2)
        fpyramid3 = self.deconv[3](fpyramid3)
        x = torch.cat([fpyramid0, fpyramid1, fpyramid2, fpyramid3], dim=1)
        x = self.conv2(x)
        return x


class ContextNetFusion(nn.Module):
    """融合全分辨率fmap1和fmap2作为corr volume"""

    def __init__(self, output_dim):
        super(ContextNetFusion, self).__init__()
        self.output_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), self.relu,
        )
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        ])
        self.fusion = nn.Conv2d(32 * 4, output_dim, 3, padding=1)

    def forward(self, fmap1, layer1, layer2, layer3):
        fmap1 = self.upsample[2](self.conv4(fmap1))
        layer3 = self.upsample[1](self.conv3(layer3))
        layer2 = self.upsample[0](self.conv2(layer2))
        layer1 = self.conv1(layer1)
        x = torch.cat([fmap1, layer3, layer2, layer1], dim=1)  # (B, 64, H, W)
        x = self.fusion(x)
        return x


class CorrBlockFR_1(nn.Module):
    """融合全分辨率fmap1和fmap2作为corr volume"""

    def __init__(self, output_dim):
        super(CorrBlockFR_1, self).__init__()
        self.output_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1), self.relu,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), self.relu,
        )
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        ])
        self.fusion = nn.Conv2d(32 * 4, output_dim, 3, padding=1)

    def forward(self, fmap1, fmap2, layer1, layer2, layer3):
        x = torch.cat([fmap1, fmap2], dim=1)
        x = self.upsample[2](self.conv4(x))
        layer3 = self.upsample[1](self.conv3(layer3))
        layer2 = self.upsample[0](self.conv2(layer2))
        layer1 = self.conv1(layer1)
        x = torch.cat([x, layer3, layer2, layer1], dim=1)  # (B, 64, H, W)
        x = self.fusion(x)
        return x


class CorrBlockFR_2(nn.Module):
    def __init__(self):
        super(CorrBlockFR_2, self).__init__()

        # self.cus_conv1 = nn.Conv2d(64, 16, (3,3), padding=1)
        # self.cus_conv1.weight.data = torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
        # self.cus_conv1.requires_grad = False
        #
        # self.cus_conv2 = nn.Conv2d(64, 16, (3, 3), padding=1)
        # self.cus_conv2.weight.data = torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
        # self.cus_conv2.requires_grad = False

        self.cus_convs = nn.ModuleList([
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.Conv2d(64, 32, 3, padding=3, dilation=3),
        ])

        for i in range(4):
            self.cus_convs[i].requires_grad = False

        # self.num_levels = num_levels
        # self.radius = radius
        # self.corr_pyramid = []
        # batch, channel, h1, w1 = corr.shape
        # corr = corr.reshape(batch*channel, 1, h1, w1)
        #
        # self.corr_pyramid.append(corr)
        # for i in range(self.num_levels-1):
        #     corr = F.avg_pool2d(corr, 2, stride=2)
        #     self.corr_pyramid.append(corr)

    # def __call__(self, coords):
    #     r = self.radius
    #     coords = coords.permute(0, 2, 3, 1)  # 代表的是flow，该怎么处理h1，w1和channel的问题
    #     batch, h1, w1, _ = coords.shape
    #
    #     out_pyramid = []
    #     for i in range(self.num_levels):
    #         corr = self.corr_pyramid[i]
    #         dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
    #         dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
    #         delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
    #
    #         centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
    #         delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
    #         coords_lvl = centroid_lvl + delta_lvl
    #
    #         corr = bilinear_sampler(corr, coords_lvl)
    #         corr = corr.view(batch, h1, w1, -1)
    #         out_pyramid.append(corr)
    #
    #     out = torch.cat(out_pyramid, dim=-1)  # out: torch.Size([1,69,75,324])
    #     return out.permute(0, 3, 1, 2).contiguous().float()  # conti()把tensor变成在内存中连续分布的形式

    def forward(self, corr):
        y = []
        for i in range(4):
            x = self.cus_convs[i](corr)
            y.append(x)
        out = torch.cat(y, dim=1)
        return out


class FlowDecoder(nn.Module):
    def __init__(self, output_dim=2, dropout=0.0):
        super(FlowDecoder, self).__init__()

        self.conv0_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.norm0_1 = nn.BatchNorm2d(256)
        self.conv0_2 = nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        self.norm0_2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv1_1 = nn.Conv2d(128 * 3, 128, kernel_size=1, stride=1)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.norm2 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 96, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.norm3 = nn.BatchNorm2d(96)
        self.conv2_1 = nn.Conv2d(96 * 3, 96, kernel_size=1)
        self.conv2_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1, stride=1)

        self.deconv3 = nn.ConvTranspose2d(96, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64 * 3, 64, kernel_size=1)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

        # output conv - output_dim = 2
        # self.se_block = SELayer(64)
        # self.conv4 = nn.Conv2d(64, output_dim, kernel_size=1)
        # self.norm5 = nn.BatchNorm2d(2)
        # 添加了一层layer0，如果效果不好，此处可以删去
        # self.conv_l0_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.conv_l0_2 = nn.Conv2d(64, 32, kernel_size=1)
        # self.norm_l0 = nn.BatchNorm2d(32)
        self.se_block = SELayer(64)

        self.conv4 = nn.Conv2d(64, output_dim, kernel_size=3, padding=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fmap1, fmap2, cnet, layer1, layer2, layer3, layer1_fb, layer2_fb, layer3_fb):

        x = torch.cat([fmap1, fmap2], dim=1)
        x = self.relu(self.norm0_1(self.conv0_1(x)))

        x = torch.cat([x, cnet], dim=1)
        x = self.relu(self.norm0_2(self.conv0_2(x)))

        x = self.relu(self.norm1(self.deconv1(x)))
        x = torch.cat([x, layer3, layer3_fb], dim=1)  # 128+128+128
        x = self.relu(self.norm1(self.conv1_1(x)))
        x = self.relu(self.norm2(self.conv1_2(x)))

        x = self.relu(self.norm3(self.deconv2(x)))
        x = torch.cat([x, layer2, layer2_fb], dim=1)  # 96 * 3
        x = self.relu(self.norm3(self.conv2_1(x)))
        x = self.relu(self.norm3(self.conv2_2(x)))

        x = self.relu(self.norm4(self.deconv3(x)))
        x = torch.cat([x, layer1, layer1_fb], dim=1)  # 64 * 3
        x = self.relu(self.norm4(self.conv3_1(x)))
        x = self.relu(self.norm4(self.conv3_2(x)))

        # 添加的layer0层，注意需要时可以删除
        # x = self.relu(self.norm_l0(self.conv_l0_1(x)))
        # x = torch.cat([x, layer0], dim=1)
        # x = self.relu(self.norm_l0(self.conv_l0_2(x)))
        x = self.se_block(x)

        # x = self.relu(self.norm5(self.conv4(x)))  # module26
        x = self.conv4(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        return x


# ======================5.25 重新修改FeaExt，将fea-pyramid1和fea-pyramid2分开输出，注意，先不加seblock==============================
class FeaExtV2(nn.Module):
    def __init__(self, output_dim=256, norm_fn='batch', dropout=0.0):
        super(FeaExtV2, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            self.norm2 = nn.GroupNorm(num_groups=8, num_channels=96)
            self.norm3 = nn.GroupNorm(num_groups=8, num_channels=128)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(96)
            self.norm3 = nn.BatchNorm2d(128)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
            self.norm2 = nn.InstanceNorm2d(96)
            self.norm3 = nn.InstanceNorm2d(128)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), self.norm1, self.relu1,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        )  # (B, 64, H, W)
        self.conv2 = self._makereslayer(64, 96, stride=2)  # (B, 96, H/2, W/2)
        self.conv3 = self._makereslayer(96, 128, stride=2)  # (B, 128, H/4, W/4)
        self.conv4 = self._makereslayer(128, output_dim, stride=2)  # (B, 256, H/8, W/8)
        # self.conv4 = nn.Conv2d(128, output_dim, kernel_size=1) 原始网络的步骤

        # SE_block
        # self.se_block = SELayer(output_dim)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _makereslayer(self, input_dim, output_dim, stride=1):
        layer1 = ResBlock(input_dim, output_dim, self.norm_fn, stride=stride)
        return nn.Sequential(layer1)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x1_pyramid = []
        x2_pyramid = []

        layer1 = self.relu1(self.norm1(self.conv1(x)))
        layer2 = self.relu1(self.norm2(self.conv2(layer1)))
        layer3 = self.relu1(self.norm3(self.conv3(layer2)))
        layer4 = self.relu1(self.conv4(layer3))
        layers = (layer1, layer2, layer3, layer4)

        # x = self.se_block(layer4)
        if self.training and self.dropout is not None:
            layer4 = self.dropout(layer4)

        if is_list:
            for i in range(4):
                layer = torch.split(layers[i], [batch_dim, batch_dim], dim=0)
                x1_pyramid.append(layer[0])
                x2_pyramid.append(layer[1])
            return x1_pyramid, x2_pyramid
        else:
            for i in range(4):
                x1_pyramid.append(layers[i])
            return x1_pyramid


class CorrBlockFRV2_1(nn.Module):
    """融合全分辨率fmap1和fmap2作为corr volume"""

    def __init__(self, output_dim):
        super(CorrBlockFRV2_1, self).__init__()
        self.output_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), self.relu,
            nn.Conv2d(128, 16, kernel_size=3, padding=1), self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 96, kernel_size=1), self.relu,
            nn.Conv2d(96, 16, kernel_size=3, padding=1), self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1), self.relu,
            nn.Conv2d(64, 16, kernel_size=3, padding=1), self.relu,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1), self.relu,
            nn.Conv2d(32, 16, kernel_size=3, padding=1), self.relu,
        )
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        ])
        self.fusion = nn.Conv2d(16 * 4, output_dim, 3, padding=1)

    def forward(self, x1_pyramid, x2_pyramid):
        layer4 = torch.cat([x1_pyramid[3], x2_pyramid[3]], dim=1)
        layer4 = self.upsample[2](self.conv4(layer4))
        layer3 = torch.cat([x1_pyramid[2], x2_pyramid[2]], dim=1)
        layer3 = self.upsample[1](self.conv3(layer3))
        layer2 = torch.cat([x1_pyramid[1], x2_pyramid[1]], dim=1)
        layer2 = self.upsample[0](self.conv2(layer2))
        layer1 = torch.cat([x1_pyramid[0], x2_pyramid[0]], dim=1)
        layer1 = self.conv1(layer1)
        x = torch.cat([layer4, layer3, layer2, layer1], dim=1)  # (B, 64, H, W)
        x = self.fusion(x)
        return x


class CorrBlockFRV2_2(nn.Module):
    def __init__(self):
        super(CorrBlockFRV2_2, self).__init__()

        # self.cus_conv1 = nn.Conv2d(64, 16, (3,3), padding=1)
        # self.cus_conv1.weight.data = torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
        # self.cus_conv1.requires_grad = False
        #
        # self.cus_conv2 = nn.Conv2d(64, 16, (3, 3), padding=1)
        # self.cus_conv2.weight.data = torch.Tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]]])
        # self.cus_conv2.requires_grad = False

        self.cus_convs = nn.ModuleList([
            nn.Conv2d(64, 32, 1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.Conv2d(64, 32, 3, padding=3, dilation=3),
        ])

        for i in range(4):
            self.cus_convs[i].requires_grad = False

    def forward(self, corr):
        y = []
        for i in range(4):
            x = self.cus_convs[i](corr)
            y.append(x)
        out = torch.cat(y, dim=1)
        return out


class ContextNetFusionV2(nn.Module):
    """融合全分辨率fmap1和fmap2作为corr volume"""

    def __init__(self, output_dim):
        super(ContextNetFusionV2, self).__init__()
        self.output_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1), self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1), self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1), self.relu,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1), self.relu,
        )
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        ])
        self.fusion = nn.Conv2d(32 * 4, output_dim, 3, padding=1)

    def forward(self, x_pyramid):
        layer4 = self.upsample[2](self.conv4(x_pyramid[3]))
        layer3 = self.upsample[1](self.conv3(x_pyramid[2]))
        layer2 = self.upsample[0](self.conv2(x_pyramid[1]))
        layer1 = self.conv1(x_pyramid[0])
        x = torch.cat([layer4, layer3, layer2, layer1], dim=1)  # (B, 64, H, W)
        x = self.fusion(x)
        return x
