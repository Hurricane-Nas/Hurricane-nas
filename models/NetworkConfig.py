import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import sys
import itertools
import random
import numpy as np
from models.operations import*
from models.ModelBuilder import*
class NetworkConfig:
    def __init__(self,num_classes,flops_constraint=330):
        self.stage_repeats=[1,4,4,8,4,1,1,1]
        self.stage_out_channels=[16,64,160,320,640,1024,0,num_classes]

        input_channel=3
        dicts={} # layer: input_channel,output_channel,choices
        layercount=0

        for idxstage in range(len(self.stage_repeats)):
            output_channel = self.stage_out_channels[idxstage]
            if idxstage==0:
                choices=['conv3x3']
                dicts=self.addToModel(dicts,layercount,input_channel,output_channel,choices)
                input_channel=output_channel
                layercount+=1
            elif idxstage==5:
                choices=['conv1x1']
                dicts=self.addToModel(dicts,layercount,input_channel,output_channel,choices)
                input_channel=output_channel
                layercount+=1
            elif idxstage==6:
                choices=['GAP']
                dicts=self.addToModel(dicts,layercount,input_channel,output_channel,choices)
                input_channel=self.stage_out_channels[5]
                layercount+=1
            elif idxstage==7:
                choices=['fc']
                dicts=self.addToModel(dicts,layercount,input_channel,output_channel,choices)
                input_channel=output_channel
                layercount+=1
            else:
                numrepeat = self.stage_repeats[idxstage]
                for i in range(numrepeat):
                    dicts=self.addToModel(dicts,layercount,input_channel,output_channel,[])
                    input_channel = output_channel
                    layercount+=1




        self.mblocks=dicts
    def addToModel(self,dicts,layercount,input_channel,output_channel,choices):
            dicts[layercount]={}
            dicts[layercount]['in_channels']=input_channel
            dicts[layercount]['out_channels']=output_channel
            dicts[layercount]['ops']=choices
            return dicts

    def getLayer(self,layerindex,op,rdicts):
        rdicts[layerindex]={}
        rdicts[layerindex]['in_channels']=self.mblocks[layerindex]['in_channels']
        rdicts[layerindex]['out_channels']=self.mblocks[layerindex]['out_channels']
        rdicts[layerindex]['op']=op
        return rdicts

    def build_modelconfig(self,ops):
        i=0
        rdicts={}
        #print(len(ops))
        while i<len(ops):
            index=i+1
            op=ops[i]
            rdicts=self.getLayer(index,op,rdicts)
            i+=1
        rdicts=self.getLayer(0,'conv_steam',rdicts)
        rdicts=self.getLayer(21,'conv1x1',rdicts)
        rdicts=self.getLayer(22,'GAP',rdicts)
        rdicts=self.getLayer(23,'fc',rdicts)
        return rdicts
