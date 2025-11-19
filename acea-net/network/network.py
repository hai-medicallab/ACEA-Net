import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

CHANNEL_1 = 16
CHANNEL_2 = 32
CHANNEL_3 = 64
CHANNEL_4 = 128
CHANNEL_5 = 256
CHANNEL_INT = 8
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        # self.identity = 1
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            #########################
            # CoordAtt3D(ch_out, ch_out),
            #########################
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),

        )
        # self.CoordAtt3D()
        self.sim = simam_module(channels=ch_out)
        self.acn = ACN(channels=ch_out)
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # y = self.conv(x)
        # print('666', y.shape, x.shape)
        # if self.identity:
        #     return x + y
        # else:
        #     return y

        x = self.conv(x)
        # x = self.CoordAtt3D(x)
        x = torch.mul(self.w1, x) + torch.mul(self.w2, self.sim(x)) + torch.mul(self.w3, self.acn(x))
        # x = torch.mul(self.w1, x) + torch.mul(self.w2, self.sim(x))

        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.InstanceNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )
        self.sim = simam_module(channels=ch_out)
        self.acn = ACN(channels=ch_out)
        self.w1 = nn.Parameter(torch.randn(1))
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.up(x)
        x = torch.mul(self.w1, x) + torch.mul(self.w2, self.sim(x)) + torch.mul(self.w3, self.acn(x))
        # x = self.sim(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.InstanceNorm3d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

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

class h_sigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt3D, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        # print(inp, oup)
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # print(x.shape)
        identity = x

        n, c, h, w, d = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2, 4)
        x_d = self.pool_d(x).permute(0, 1, 4, 3, 2)

        # print('111', x_h.shape, x_w.shape)
        y = torch.cat([x_h, x_w, x_d], dim=2)
        # y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # print('3333', y.shape)

        x_h, x_w, x_d = torch.split(y, [x_h.shape[2], x_w.shape[2], x_d.shape[2]], dim=2)
        # x_h, x_w = torch.split(y, [x_h.shape[2], x_w.shape[2]], dim=2)
        # x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2, 4)
        x_d = x_d.permute(0, 1, 4, 3, 2)
        # print('555', x_h.shape, x_w.shape, x_d.shape)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_d = self.conv_d(x_d).sigmoid()
        a_h = a_h.expand(-1, -1, h, w, d)
        a_w = a_w.expand(-1, -1, h, w, d)
        a_d = a_d.expand(-1, -1, h, w, d)
        # print('555', a_h.shape, a_w.shape, a_d.shape)

        out = identity * a_d * a_h * a_w
        # out = identity * a_h * a_w
        # print('777', out.shape)

        return out

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

class U_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(U_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=CHANNEL_1)
        self.Conv2 = conv_block(ch_in=CHANNEL_1, ch_out=CHANNEL_2)
        self.Conv3 = conv_block(ch_in=CHANNEL_2, ch_out=CHANNEL_3)
        self.Conv4 = conv_block(ch_in=CHANNEL_3, ch_out=CHANNEL_4)
        self.Conv5 = conv_block(ch_in=CHANNEL_4, ch_out=CHANNEL_5)

        self.Up5 = up_conv(ch_in=CHANNEL_5, ch_out=CHANNEL_4)
        self.Up_conv5 = conv_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4)

        self.Up4 = up_conv(ch_in=CHANNEL_4, ch_out=CHANNEL_3)
        self.Up_conv4 = conv_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3)
        
        self.Up3 = up_conv(ch_in=CHANNEL_3, ch_out=CHANNEL_2)
        self.Up_conv3 = conv_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2)
        
        self.Up2 = up_conv(ch_in=CHANNEL_2, ch_out=CHANNEL_1)
        self.Up_conv2 = conv_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1, output_ch, kernel_size=1, stride=1, padding=0)

###################################################################################
        self.maxpool = torch.nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.avgpool = torch.nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

        self.con1_f234 = nn.Conv3d(224, 32, kernel_size=1, bias=True)
        self.con2_f234 = nn.Conv3d(32, 16, kernel_size=1, bias=True)
        self.con3_f1234 = nn.Conv3d(32, 16, kernel_size=1, bias=True)
        self.con4_f1234 = nn.Conv3d(16, 16, kernel_size=1, bias=True)
        self.con_f1234 = nn.Conv3d(240, 4, kernel_size=1, bias=True)
        self.conv3_3_3 = nn.Conv3d(4, 4, kernel_size=3, padding='same', bias=True)
        self.conv1_1_1 = nn.Conv3d(4, 4, kernel_size=1, bias=True)
        self.conv1_1_1_1 = nn.Conv3d(4, 16, kernel_size=1, bias=True)

        self.acn_x2 = ACN(CHANNEL_2)  # 添加 ACN 层
        self.acn_x3 = ACN(CHANNEL_3)  # 添加 ACN 层到 d3
        self.acn_x4 = ACN(CHANNEL_4)  # 添加 ACN 层到 d4
        self.acn_x5 = ACN(CHANNEL_5)

        self.acn_d2 = ACN(CHANNEL_1)  # 添加 ACN 层
        self.acn_d3 = ACN(CHANNEL_2)  # 添加 ACN 层到 d3
        self.acn_d4 = ACN(CHANNEL_3)  # 添加 ACN 层到 d4
        self.acn_d5 = ACN(CHANNEL_4)

        self.CoordAtt3D_1 = CoordAtt3D(CHANNEL_1, CHANNEL_1)
        self.CoordAtt3D_2 = CoordAtt3D(CHANNEL_2, CHANNEL_2)
        self.CoordAtt3D_3 = CoordAtt3D(CHANNEL_3, CHANNEL_3)
        self.CoordAtt3D_4 = CoordAtt3D(CHANNEL_4, CHANNEL_4)
        self.CoordAtt3D_5 = CoordAtt3D(CHANNEL_5, CHANNEL_5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # encoding path
        x1 = self.Conv1(x)
        # print('111', x1)
        # x1 = self.CoordAtt3D_1(x1)
        # print('222', x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        # x2 = self.acn_x2(x2)
        # x2 = x2 + self.CoordAtt3D_2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        # x3 = self.acn_x3(x3)
        # x3 = x3 + self.CoordAtt3D_3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # x4 = self.acn_x4(x4)
        # x4 = x4 + self.CoordAtt3D_4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # x5 = self.acn_x5(x5)
        # x5 = x5 + self.CoordAtt3D_5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)  # f1
        # d5 = self.acn_d5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)  # f2
        # d4 = self.acn_d4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)   # f3
        # d3 = self.acn_d3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)  # f4
        # d2 = self.acn_d2(d2)
        # print("2", d2.shape,d3.shape,d4.shape,d5.shape)
        # torch.Size([1, 16, 128, 128, 48])
        # torch.Size([1, 32, 64, 64, 24])
        # torch.Size([1, 64, 32, 32, 12])
        # torch.Size([1, 128, 16, 16, 6])

############################################################################################
        #
        # #    d2     # torch.Size([1, 128, 16, 16, 6])
        # #    d3     # torch.Size([1, 64, 32, 32, 12])
        # #    d4     # torch.Size([1, 32, 64, 64, 24])
        # #    d5     # torch.Size([1, 16, 128, 128, 48])
        # d21 = d2
        # d2 = d5
        # d5 = d21
        # d31 = d3
        # d3 = d4
        # d4 = d31
        #
        # # print("2", d2.shape, d3.shape, d4.shape, d5.shape)
        # d2 = F.interpolate(d2, size=(32, 32, 12), mode='trilinear', align_corners=False)
        # f34 = torch.cat((d3, d2), 1)  # ([1, 96, 32, 32, 12])
        # f34 = F.interpolate(f34, size=(64, 64, 24), mode='trilinear', align_corners=False)  # [1, 96, 64, 64, 24]
        # f234 = torch.cat((d4, f34), 1)  # # ([1, 224, 64, 64, 24])
        #
        # f234 = self.con1_f234(f234) # [1, 32, 64, 64, 24]
        # f234 = torch.sigmoid(f234)
        # f234 = torch.mul(f234, d4)  # ([1, 32, 64, 64, 24])
        #
        # f234 = self.con2_f234(f234)
        # f234 = F.interpolate(f234, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f1234 = torch.cat((d5, f234), 1)  # ([1, 32, 128, 128, 48])
        #
        # f1234 = self.con3_f1234(f1234)  # [1, 16, 128, 128, 48]
        # f1234 = torch.sigmoid(f1234)
        # f1234 = torch.mul(f1234, d5)  # ([1, 16, 128, 128, 48])
        #
        # f1234 = self.con4_f1234(f1234)  # ([1, 16, 128, 128, 48])
        # # f1234 = self.Up_f1234(f1234)
        #
        # d2 = F.interpolate(d2, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # d3 = F.interpolate(d3, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # d4 = F.interpolate(d4, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f_1234 = torch.cat((d5, d4, d3, d2), 1)
        #
        # f_1234 = self.con_f1234(f_1234)  # [1, 4, 128, 128, 48]
        # # print('33', f_1234.shape)
        # f_1234_1 = torch.sigmoid(torch.add(self.maxpool(f_1234), self.avgpool(f_1234)))
        # f_1234_1 = F.interpolate(f_1234_1, size=(128, 128, 48), mode='trilinear', align_corners=False)
        # f_1234_1 = torch.mul(f_1234, f_1234_1)   # [1, 4, 128, 128, 48]
        #
        #
        #
        #
        # f_1234_1_1 = self.conv3_3_3(f_1234_1)
        #
        # # conv1_1_1.to("cuda")
        # f_1234_1_2 = self.conv1_1_1(f_1234_1_1)
        #
        # f_1234_1_2 = torch.sigmoid(f_1234_1_2)
        # f_1234_1_1 = torch.mul(f_1234_1_1, f_1234_1_2)
        #
        #
        # f_1234_1 = torch.add(f_1234_1_1, f_1234_1)
        # f_1234 = torch.add(f_1234, f_1234_1)
        # # conv1_1_1_1.to("cuda")
        # f_1234 = self.conv1_1_1_1(f_1234)
        # x = torch.add(f_1234, f1234)
        #
        # d2 = x
        # # print('11', x.shape)    # [1,16,128,128,48]

        d1 = self.Conv_1x1(d2)  # f~
        # d1 = self.sigmoid(d1)

        # print('555', d1)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=CHANNEL_1,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=CHANNEL_1,ch_out=CHANNEL_2,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=CHANNEL_2,ch_out=CHANNEL_3,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=CHANNEL_3,ch_out=CHANNEL_4,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=CHANNEL_4,ch_out=CHANNEL_5,t=t)
        

        self.Up5 = up_conv(ch_in=CHANNEL_5,ch_out=CHANNEL_4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4,t=t)
        
        self.Up4 = up_conv(ch_in=CHANNEL_4,ch_out=CHANNEL_3)
        self.Up_RRCNN4 = RRCNN_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3,t=t)
        
        self.Up3 = up_conv(ch_in=CHANNEL_3,ch_out=CHANNEL_2)
        self.Up_RRCNN3 = RRCNN_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2,t=t)
        
        self.Up2 = up_conv(ch_in=CHANNEL_2,ch_out=CHANNEL_1)
        self.Up_RRCNN2 = RRCNN_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1,t=t)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=CHANNEL_1)
        self.Conv2 = conv_block(ch_in=CHANNEL_1,ch_out=CHANNEL_2)
        self.Conv3 = conv_block(ch_in=CHANNEL_2,ch_out=CHANNEL_3)
        self.Conv4 = conv_block(ch_in=CHANNEL_3,ch_out=CHANNEL_4)
        self.Conv5 = conv_block(ch_in=CHANNEL_4,ch_out=CHANNEL_5)

        self.Up5 = up_conv(ch_in=CHANNEL_5,ch_out=CHANNEL_4)
        self.Att5 = Attention_block(F_g=CHANNEL_4,F_l=CHANNEL_4,F_int=CHANNEL_3)
        self.Up_conv5 = conv_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4)

        self.Up4 = up_conv(ch_in=CHANNEL_4,ch_out=CHANNEL_3)
        self.Att4 = Attention_block(F_g=CHANNEL_3,F_l=CHANNEL_3,F_int=CHANNEL_2)
        self.Up_conv4 = conv_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3)
        
        self.Up3 = up_conv(ch_in=CHANNEL_3,ch_out=CHANNEL_2)
        self.Att3 = Attention_block(F_g=CHANNEL_2,F_l=CHANNEL_2,F_int=CHANNEL_1)
        self.Up_conv3 = conv_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2)
        
        self.Up2 = up_conv(ch_in=CHANNEL_2,ch_out=CHANNEL_1)
        self.Att2 = Attention_block(F_g=CHANNEL_1,F_l=CHANNEL_1,F_int=CHANNEL_INT)
        self.Up_conv2 = conv_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=CHANNEL_1,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=CHANNEL_1,ch_out=CHANNEL_2,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=CHANNEL_2,ch_out=CHANNEL_3,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=CHANNEL_3,ch_out=CHANNEL_4,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=CHANNEL_4,ch_out=CHANNEL_5,t=t)
        

        self.Up5 = up_conv(ch_in=CHANNEL_5,ch_out=CHANNEL_4)
        self.Att5 = Attention_block(F_g=CHANNEL_4,F_l=CHANNEL_4,F_int=CHANNEL_3)
        self.Up_RRCNN5 = RRCNN_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4,t=t)
        
        self.Up4 = up_conv(ch_in=CHANNEL_4,ch_out=CHANNEL_3)
        self.Att4 = Attention_block(F_g=CHANNEL_3,F_l=CHANNEL_3,F_int=CHANNEL_2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3,t=t)
        
        self.Up3 = up_conv(ch_in=CHANNEL_3,ch_out=CHANNEL_2)
        self.Att3 = Attention_block(F_g=CHANNEL_2,F_l=CHANNEL_2,F_int=CHANNEL_1)
        self.Up_RRCNN3 = RRCNN_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2,t=t)
        
        self.Up2 = up_conv(ch_in=CHANNEL_2,ch_out=CHANNEL_1)
        self.Att2 = Attention_block(F_g=CHANNEL_1,F_l=CHANNEL_1,F_int=CHANNEL_INT)
        self.Up_RRCNN2 = RRCNN_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1,t=t)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
