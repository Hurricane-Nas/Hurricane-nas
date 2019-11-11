import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import torch.nn.functional as F
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MBInvertedConvLayer(nn.Module):
    def __init__(self,kernel_size,in_channels,out_channels,stride,expand_ratio=6,se=False,se_paras='s_4',ratio=1.0):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        if kernel_size == 1:
          pad = 0
        elif kernel_size == 3:
          pad = 1
        elif kernel_size == 5:
          pad = 2
        elif kernel_size == 7:
          pad = 3
        feature_dim=round(in_channels * self.expand_ratio)
        middle_inc=int(feature_dim*ratio )
        if middle_inc%2!=0:
            middle_inc+=1
        #print(feature_dim,middle_inc)
        if se==False:
            self.op=nn.Sequential(
            nn.Conv2d(in_channels, middle_inc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(middle_inc),
            nn.ReLU6(inplace=True),
            nn.Conv2d(middle_inc,middle_inc,kernel_size,stride,pad,groups=middle_inc,bias=False),
            nn.BatchNorm2d(middle_inc),
            nn.ReLU6(inplace=True),
            nn.Conv2d(middle_inc, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            )
        else:
            self.op=nn.Sequential(
            nn.Conv2d(in_channels, middle_inc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(middle_inc),
            nn.ReLU6(inplace=True),
            nn.Conv2d(middle_inc,middle_inc,kernel_size,stride,pad,groups=middle_inc,bias=False),
            nn.BatchNorm2d(middle_inc),
            SEModule(middle_inc,se_paras),
            nn.ReLU6(inplace=True),
            nn.Conv2d(middle_inc, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        if self.stride==2:
            x = self.op(x)
        else:
            x1=x
            x2=self.op(x)
            x=torch.add(x1,x2)
        return x

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x




class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, se_paras='hsx4'):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        act=se_paras.split('x')[0]
        reduction=int(se_paras.split('x')[1])
        if act=='hs':
            self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )
        if act=='s':

                self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel, bias=False),
                #Hsigmoid()
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out= x * y.expand_as(x)
       # print('se', x.shape,'out',out.shape)
        return out
class SepConv(nn.Module):

  def __init__(self,  kernel_size, C_in, C_out,stride,se=False,se_paras='s_4',ratio=1.0):
    super(SepConv, self).__init__()
    if kernel_size==3:
        padding=1
    if kernel_size==5:
        padding=2
    if kernel_size==7:
        padding=3
    middle_inc=int(C_in*ratio)
    if middle_inc%2>0:
        middle_inc+=1
    #print('middle_inc',middle_inc)
    if se==False:
        self.op = nn.Sequential(

      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, middle_inc, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(middle_inc),
      nn.ReLU(inplace=True),
      nn.Conv2d(middle_inc, middle_inc, kernel_size=kernel_size, stride=1, padding=padding, groups=middle_inc, bias=False),
      nn.Conv2d(middle_inc, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out),
      nn.ReLU(inplace=True),
      )
    else:
      self.op = nn.Sequential(

      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, middle_inc, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(middle_inc),
      nn.ReLU(inplace=True),
      nn.Conv2d(middle_inc, middle_inc, kernel_size=kernel_size, stride=1, padding=padding, groups=middle_inc, bias=False),
      nn.Conv2d(middle_inc, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out),
      nn.ReLU(inplace=True),
      SEModule(C_out,se_paras),
    )

  def forward(self, x):
    #print(x.shape)
    return self.op(x)


class Choice_x(nn.Module):
    def __init__(self, kernelsize,inp, oup, stride,se=False,se_paras='s_4',ratio=1.0):
        super(Choice_x, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        middle_inc=int(oup_inc*ratio)
        if middle_inc%2>0:
            middle_inc+=1
        #print('middle_inc',middle_inc)
        if self.stride == 1:
            #assert inp == oup_inc
            if se==False:
            	self.banch2 = nn.Sequential(
                    # dw
                    nn.Conv2d(inp//2, inp//2, 3, stride, 1,groups=inp//2, bias=False),
                    nn.BatchNorm2d(inp//2),
                    # pw
                    nn.Conv2d(inp//2, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_inc,middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),

                )
            else:
                self.banch2 = nn.Sequential(
                    # dw
                    nn.Conv2d(inp//2, inp//2, 3, stride, 1,groups=inp//2, bias=False),
                    nn.BatchNorm2d(inp//2),
                    # pw
                    nn.Conv2d(inp//2, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_inc, middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                    SEModule(oup_inc,se_paras),

                )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernelsize, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),

            )
            if se==False:
                self.banch2 = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, 1, 1,groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    # pw
                    nn.Conv2d(inp, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(middle_inc, middle_inc, 3, 1, 1,groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),

                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),

                )
            else:
                    self.banch2 = nn.Sequential(
                        nn.Conv2d(inp, inp, 3, 1, 1,groups=inp, bias=False),
                        nn.BatchNorm2d(inp),
                        # pw
                        nn.Conv2d(inp, middle_inc, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(middle_inc),
                        nn.ReLU(inplace=True),
                        # dw
                        nn.Conv2d(middle_inc, middle_inc, 3, stride, 1,groups=middle_inc, bias=False),
                        nn.BatchNorm2d(middle_inc),

                        nn.Conv2d(middle_inc, middle_inc, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(middle_inc),
                        nn.ReLU(inplace=True),

                        nn.Conv2d(middle_inc, middle_inc, 3, 1, 1,groups=middle_inc, bias=False),
                        nn.BatchNorm2d(middle_inc),

                        nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(oup_inc),
                        nn.ReLU(inplace=True),
                        SEModule(oup_inc,se_paras),

                    )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        #print('x1',x.shape,'out',out.shape)
        return torch.cat((x, out), 1)

    def forward(self, x):
        #print('input shape',x.shape)
        if 1==self.stride:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            #print('x1',x1.shape,'x2',x2.shape)
            out = self._concat(x1, self.banch2(x2))

        elif 2==self.stride:
            out = self._concat(self.banch1(x), self.banch2(x))
        #print('output shape',x.shape)
        return channel_shuffle(out, 2)

class InvertedResidual(nn.Module):
    def __init__(self, kernelsize,inp, oup, stride,se=False,se_paras='s_4',ratio=1.0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        middle_inc=int(oup_inc*ratio)
        if middle_inc%2>0:
            middle_inc+=1

        #print('middle_inc',middle_inc)
        if kernelsize==3:
            pad=1
        if kernelsize==5:
            pad=2
        if kernelsize==7:
            pad=3

        if self.stride == 1:
            #assert inp == oup_inc
            if se==False:
            	self.banch2 = nn.Sequential(
                    # pw
                    nn.Conv2d(inp//2, middle_inc, 1, 1, 0, bias=False), ## when stride=1, in_channels=out_channels, thus, after channel split, inp_inc=oup_inc
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, kernelsize, stride, pad, groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    # pw-linear
                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                )
            else:
                self.banch2 = nn.Sequential(
                    # pw
                    nn.Conv2d(inp//2, middle_inc, 1, 1, 0, bias=False), ## when stride=1, in_channels=out_channels, thus, after channel split, inp_inc=oup_inc
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, kernelsize, stride, pad, groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    # pw-linear
                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                    SEModule(oup_inc,se_paras),
                )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernelsize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
            if se==False:
                self.banch2 = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, kernelsize, stride, pad, groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    # pw-linear
                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                )
            else:
                self.banch2 = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, middle_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    nn.ReLU(inplace=True),
                    # dw
                    nn.Conv2d(middle_inc, middle_inc, kernelsize, stride, pad, groups=middle_inc, bias=False),
                    nn.BatchNorm2d(middle_inc),
                    # pw-linear
                    nn.Conv2d(middle_inc, oup_inc, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup_inc),
                    nn.ReLU(inplace=True),
                    SEModule(oup_inc,se_paras),
                )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        #print('x1',x.shape,'out',out.shape)
        return torch.cat((x, out), 1)

    def forward(self, x):
        #print('input shape',x.shape)
        if 1==self.stride:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            #print('x1',x1.shape,'x2',x2.shape)
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.stride:
            out = self._concat(self.banch1(x), self.banch2(x))
        #print('output shape',x.shape)
        return channel_shuffle(out, 2)
