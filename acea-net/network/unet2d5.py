#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from torch import nn
import torch
import numpy as np
import torch.nn.functional
import torch.nn.functional as F

class ACN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(ACN, self).__init__()
        self.bn = nn.BatchNorm3d(channels, eps=eps)
        # self.bn = nn.InstanceNorm3d(channels, eps=eps)

        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

        self.eps = eps  # 添加 eps 作为实例变量
    def forward(self, x):
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True)

        norm_x = (x - mean) / (var + self.eps).sqrt()
        y = self.gamma * norm_x + self.beta
        return y

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w, d = x.size()

        n = w * h * d - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5
        y = self.activaton(y)
        # print('111', y, y.shape)
        out = x * y
        return out

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvNormNonlinBlock(nn.Module):   # 卷积归一化激活*2
    def __init__(
        self, 
        input_channels, 
        output_channels,
        conv_op=nn.Conv3d, 
        conv_kwargs=None,
        norm_op=nn.InstanceNorm3d,  # 三维的归一化
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs=None):

        """
        Block: Conv->Norm->Activation->Conv->Norm->Activation
        """

        super(ConvNormNonlinBlock, self).__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.output_channels = output_channels

        self.first_conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.first_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.first_acti = self.nonlin(**self.nonlin_kwargs)

        self.second_conv = self.conv_op(output_channels, output_channels, **self.conv_kwargs)
        self.second_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.second_acti = self.nonlin(**self.nonlin_kwargs)        

        self.block = nn.Sequential(
            self.first_conv,
            self.first_norm,
            self.first_acti,
            self.second_conv,
            self.second_norm,
            self.second_acti
            )
#############################################################
        self.sim = simam_module(channels=output_channels)
        self.acn = ACN(channels=output_channels)
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x):

        x = self.block(x)

        # x = torch.mul(self.w1, x) + torch.mul(self.w2, self.sim(x)) + torch.mul(self.w3, self.acn(x))
        # x = torch.mul(self.w1, x) + torch.mul(self.w2, self.sim(x))
        # x = torch.mul(self.w1, x) + torch.mul(self.w3, self.acn(x))

        return x



class Upsample(nn.Module):  # 上采样
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor,
                                         mode=self.mode, align_corners=self.align_corners)


class U_Net2D5(nn.Module):

    def __init__(
        self, 
        input_channels=1, 
        base_num_features=16, 
        num_classes=2, 
        num_pool=4,
        conv_op=nn.Conv3d,
        conv_kernel_sizes=None,
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        weightInitializer=InitWeights_He(1e-2)):
        """
        2.5D CNN combining 2D and 3D convolutions to dealwith the low through-plane resolution.
        The first two stages have 2D convolutions while the others have 3D convolutions. 

        Architecture inspired by: 
        Wang,et al: Automatic segmentation of  vestibular  schwannoma  from  t2-weighted  mri  
        by  deep  spatial  attention  with hardness-weighted loss. MICCAI 2019. 
        """
        super(U_Net2D5, self).__init__()

        if nonlin_kwargs is None:
             nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias':True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_classes = num_classes

        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 1)] * 2 + [(3, 3, 3)]*(num_pool - 1)

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []

        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        input_features = input_channels
        output_features = base_num_features

        for d in range(num_pool):
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, output_features,
                                                                self.conv_op, self.conv_kwargs, self.norm_op,
                                                                self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))

            self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = 2 * output_features  # Number of kernel increases by a factor 2 after each pooling

        final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, final_num_features,
                                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                                            self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u+1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u+1)]
            self.conv_blocks_localization.append(ConvNormNonlinBlock(n_features_after_tu_and_concat, final_num_features,
                                                 self.conv_op, self.conv_kwargs, self.norm_op,
                                                 self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))

        self.final_conv = conv_op(self.conv_blocks_localization[-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)


##################################################################
        # self.maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        # self.avgpool = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)
        # self.con1_f234 = nn.Conv3d(112,16 ,kernel_size=1, bias=True)
        # self.con2_f234 = nn.Conv3d(16, 16, kernel_size=1, bias=True)
        # self.con3_f1234 = nn.Conv3d(32, 16, kernel_size=1, bias=True)
        # self.con4_f1234 = nn.Conv3d(16, 16, kernel_size=1, bias=True)
        # self.con_f1234 = nn.Conv3d(128, 4, kernel_size=1, bias=True)
        # self.conv3_3_3 = nn.Conv3d(4, 4, kernel_size=3, padding='same', bias=True)
        # self.conv1_1_1 = nn.Conv3d(4, 4, kernel_size=1, bias=True)
        # self.conv1_1_1_1 = nn.Conv3d(4, 16, kernel_size=1, bias=True)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        # f_ = []
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        #     f_.append(x)
        # d2 = f_[0]  # ([1, 64, 16, 16, 6])
        # d3 = f_[1]  # ([1, 32, 32, 32, 12])
        # d4 = f_[2]  # ([1, 16, 64, 64, 24])
        # d5 = f_[3]  # ([1, 16, 128, 128, 48])
        # d2 = F.interpolate(d2, size=(32, 32, 12), mode='trilinear', align_corners=False)
        # f34 = torch.cat((d3, d2), 1)  # ([1, 96, 32, 32, 12])
        # f34 = F.interpolate(f34, size=(64, 64, 24), mode='trilinear', align_corners=False)  # [1, 96, 64, 64, 24]
        # f234 = torch.cat((d4, f34), 1)  # # ([1, 112, 64, 64, 24])
        # # con1_f234.to("cuda")  # device是你的CUDA设备
        # f234 = self.con1_f234(f234)  # [1, 16, 64, 64, 24]
        # # print('22', f234.shape)
        # f234 = F.sigmoid(f234)
        # f234 = torch.mul(f234, d4)  # ([1, 16, 64, 64, 24])
        # # con2_f234.to("cuda")  # device是你的CUDA设备
        # f234 = self.con2_f234(f234)
        # f234 = F.interpolate(f234, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f1234 = torch.cat((d5, f234), 1)  # ([1, 16, 128, 128, 48])
        # # con3_f1234.to("cuda")  # device是你的CUDA设备
        # f1234 = self.con3_f1234(f1234)  # [1, 16, 128, 128, 48]
        # f1234 = F.sigmoid(f1234)
        # f1234 = torch.mul(f1234, d5)
        # # con4_f1234.to("cuda")  # device是你的CUDA设备
        # f1234 = self.con4_f1234(f1234)
        # # f1234 = self.Up_f1234(f1234)
        #
        #
        # d2 = F.interpolate(d2, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # d3 = F.interpolate(d3, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # d4 = F.interpolate(d4, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f_1234 = torch.cat((d5, d4, d3, d2), 1)
        # # con_f1234.to("cuda")  # device是你的CUDA设备
        # f_1234 = self.con_f1234(f_1234)  # [1, 4, 128, 128, 48]
        # # print('33', f_1234.shape)
        # f_1234_1 = F.sigmoid(torch.add(self.maxpool(f_1234), self.avgpool(f_1234)))
        # f_1234_1 = F.interpolate(f_1234_1, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f_1234_1 = torch.mul(f_1234, f_1234_1)
        #
        #
        #
        # # conv3_3_3.to("cuda")  # device是你的CUDA设备
        # f_1234_1_1 = self.conv3_3_3(f_1234_1)
        #
        # # conv1_1_1.to("cuda")
        # f_1234_1_2 = self.conv1_1_1(f_1234_1_1)
        #
        # f_1234_1_2 = F.sigmoid(f_1234_1_2)
        # f_1234_1_1 = torch.mul(f_1234_1_1, f_1234_1_2)
        #
        #
        # f_1234_1 = torch.add(f_1234_1_1, f_1234_1)
        # f_1234 = torch.add(f_1234, f_1234_1)
        # # conv1_1_1_1.to("cuda")
        # f_1234 = self.conv1_1_1_1(f_1234)
        # f = torch.add(f_1234, f1234)
        #
        # # print('11', x.shape)    # [1,16,128,128,48]

        output = self.final_conv(x)
        return output
                




