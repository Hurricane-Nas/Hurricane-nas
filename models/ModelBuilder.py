import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import sys

from models.operations import*


class ModelBuilder(nn.Module) :
    def __init__(self,modelconfigs,n_classes,input_size=224):
        super(ModelBuilder,self).__init__()
        #print(n_classes)
        self.layers=[]
        self.conv_steam=conv_bn(3, 16, 2)
        layerindexs=list(modelconfigs.keys())
        layerindexs.sort()

        for layerindex in layerindexs:
            if layerindex not in [0,21,22,23]:
                layer=self.getLayer(modelconfigs[layerindex])
            #print(layerindex,layer)
                self.layers.append(layer)

        self.features=nn.Sequential(*self.layers)
        self.conv1x1= conv_1x1_bn(640, 1024)
        self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(nn.Linear(1024, n_classes))


    def forward(self,x):
        #print(x.shape)
        x = self.conv_steam(x)
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x = self.conv1x1(x)
        #print(x.shape)
        x = self.globalpool(x)
        #x = x.view(x.size(0),-1)
        #print(x.shape)
        #print(x.shape)
        #print(x)
        x=self.dropout(x)
        x=x.view(-1, 1024)
        #print('after x',x.shape)
        #print(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x
    def getLayer(self,config):
        c_in=config["in_channels"]
        c_out=config["out_channels"]
        op=config["op"]
        #print('current op',op,c_in,c_out)
        ratio=1.0
        if "xx" in op:
            ratio=float(op.split('xx')[0])
        if 'choice' in op:
            if 'choicex' not in op:
                kernel=int(op.split('_')[0].split('choice')[-1])
                stride=int(op.split('_')[1])
                se=False
                separas=""
                if '_se_' in op:
                    se=True
                    separas=op.split('_')[-1]
                return InvertedResidual(kernel,c_in,c_out,stride,se=se,se_paras=separas,ratio=ratio)
            else:
                #$kernel=int(op.split('_')[0].split('choicex')[-1])
                stride=int(op.split('_')[1])
                se=False
                separas=""
                if '_se_' in op:
                    se=True
                    separas=op.split('_')[-1]
                return Choice_x(3,c_in,c_out,stride,se=se,se_paras=separas,ratio=ratio)


        if 'sep' in op:
            kernel=int(op.split('_')[0].split('sep')[-1])
            stride=int(op.split('_')[1])
            se=False
            separas=""

            if '_se_' in op:
                se=True
                separas=op.split('_')[-1]
            return SepConv(3,c_in,c_out,stride,se=se,se_paras=separas,ratio=ratio)

        if op=='dil3_1':
            return DilConv(3,c_in,c_out,1)
        if op=='dil5_1':
            return DilConv(5,c_in,c_out,1)
        if op=='Identity':
            return Identity()
        if 'mb' in op:
            items=op.split('_')
            e=int(items[2])
            stride=int(items[1])
            kernel=int(items[0].split("mb")[-1])
            se=False
            separas=""
            if '_se_' in op:
                se=True
                separas=op.split('_')[-1]
            return MBInvertedConvLayer(kernel,c_in,c_out,stride,e,se=se,se_paras=separas,ratio=ratio)
