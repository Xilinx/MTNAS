# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from torch.autograd import Variable
from layers import *
from operations import *
import pdb

solver = {
    'k1': 8,
    'k2': 8,
    'act_clip_val': 8,
    'warmup': False,
    'det_classes': 4,
    'seg_classes': 16,
    'lr_steps': (12000, 18000),
    #'lr_steps': (5, 10),
    'max_iter': 20010,
    'feature_maps': [(80,128), (40,64), (20,32), (10,16), (5,8), (3,6), (1,4)],
    'resize': (320,512),
    'steps': [4, 8, 16, 32, 64, 128, 256],
    'min_sizes': [10, 30, 60, 100, 160, 220, 280],
    'max_sizes': [30, 60, 100, 160, 220, 280, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': False,
}
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        """

        :param genotype:
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction:
        :param reduction_prev:
        """
        super(Cell, self).__init__()

        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        """
        :param C:
        :param op_names:
        :param indices:
        :param concat:
        :param reduction:
        :return:
        """
        assert len(op_names) == len(indices)

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        """

        :param s0:
        :param s1:
        :param drop_prob:
        :return:
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)

class DetCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,reduction_prev,kernal3):
        """

        :param genotype:
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction:
        """
        super(DetCell, self).__init__()

        print(C_prev_prev, C_prev, C)

        if kernal3 == False:
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(C_prev_prev, C)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        
        if kernal3 == True:
            if reduction_prev:
                self.preprocess0 = DETFactorizedReduce(C_prev_prev, C)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev, C, 3, 1, 0)
            self.preprocess1 = ReLUConvBN(C_prev, C, 3, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        """
        :param C:
        :param op_names:
        :param indices:
        :param concat:
        :param reduction:
        :return:
        """
        assert len(op_names) == len(indices)

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        """

        :param s0:
        :param s1:
        :param drop_prob:
        :return:
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
    
class SegCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction,reduction_prev):
        """

        :param genotype:
        :param C_prev_prev:
        :param C_prev:
        :param C:
        :param reduction:
        """
        super(SegCell, self).__init__()

        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        """
        :param C:
        :param op_names:
        :param indices:
        :param concat:
        :param reduction:
        :return:
        """
        assert len(op_names) == len(indices)

        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        """

        :param s0:
        :param s1:
        :param drop_prob:
        :return:
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)

            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
    
class Network(nn.Module):
    def __init__(self, C, layers, genotype,Detgenotype,Seggenotype,det_classes, seg_classes):
        super(Network, self).__init__()
        self.num_classes = det_classes
        self.seg_classes = seg_classes
        self._layers = layers

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C
        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [6,9,12]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        
        self.detcells = nn.ModuleList()
        
        reduction_prev = False
        for i in range(3):
            if i == 0:
                reduction = True
                C_prev_prev, C_prev, C_curr = 1024, 1024, 256
                kernal3=False
            elif i ==1:
                reduction=False
                C_prev_prev, C_prev, C_curr = 1024, 1024, 128
                C_curr = 128
                kernal3= True
            else:
                reduction = False
                C_curr = 128
                C_prev_prev, C_prev, C_curr = 512, 512, 128
                kernal3= True
            detcell = DetCell(Detgenotype,C_prev_prev, C_prev, C_curr, reduction, reduction_prev,kernal3)
            reduction_prev = reduction
            self.detcells += [detcell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        
        self.segcells = nn.ModuleList()
        # pdb.set_trace()
        reduction_prev = reduction = False
        for i in range(4):
            if i ==0:
                reduction=False
                C_prev_prev, C_prev, C_curr = 1024, 1024, 128
            if i == 1:
                reduction=False
                C_prev_prev, C_prev, C_curr = 512, 512, 64
            if i == 2:
                reduction=False
                C_prev_prev, C_prev, C_curr = 256,256, 32
            if i == 3 :
                reduction=False
                C_prev_prev, C_prev, C_curr = 128 ,128, 8
            segcell = SegCell(Seggenotype,C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.segcells += [segcell]
            # C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
        
        self.score_2s = nn.Sequential(OrderedDict([
            ('score_2s_conv', nn.Conv2d(32, self.seg_classes, kernel_size=1)),
            ('score_2s_BN', nn.BatchNorm2d(self.seg_classes)), ]))

        # self.conv_block6 = nn.Sequential(OrderedDict([
        #     ('conv_block6_conv1', nn.Conv2d(1024, 512, 1, padding=0)),
        #     ('conv_block6_BN1', nn.BatchNorm2d(512)),
        #     ('conv_block6_relu1', nn.ReLU(inplace=True)),
        #     ('conv_block6_conv2', nn.Conv2d(512, 1024, 3, padding=1, stride=2)),
        #     ('conv_block6_BN2', nn.BatchNorm2d(1024)),
        #     ('conv_block6_relu2', nn.ReLU(inplace=True)), ]))

        # self.conv_block7 = nn.Sequential(OrderedDict([
        #     ('conv_block7_conv1', nn.Conv2d(1024, 256, 1, padding=0)),
        #     ('conv_block7_BN1', nn.BatchNorm2d(256)),
        #     ('conv_block7_relu1', nn.ReLU(inplace=True)),
        #     ('conv_block7_conv2', nn.Conv2d(256, 512, 3, padding=0)),
        #     ('conv_block7_BN2', nn.BatchNorm2d(512)),
        #     ('conv_block7_relu2', nn.ReLU(inplace=True)), ]))

        # self.conv_block8 = nn.Sequential(OrderedDict([
        #     ('conv_block8_conv1', nn.Conv2d(512, 256, 1, padding=0)),
        #     ('conv_block8_BN1', nn.BatchNorm2d(256)),
        #     ('conv_block8_relu1', nn.ReLU(inplace=True)),
        #     ('conv_block8_conv2', nn.Conv2d(256, 512, 3, padding=0)),
        #     ('conv_block8_BN2', nn.BatchNorm2d(512)),
        #     ('conv_block8_relu2', nn.ReLU(inplace=True)), ]))

        # self.toplayer3 = nn.Sequential(OrderedDict([
        #     ('toplayer3_conv', nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)),
        #     ('toplayer3_BN', nn.BatchNorm2d(512)), ]))  # Reduce channels
        # self.toplayer2 = nn.Sequential(OrderedDict([
        #     ('toplayer2_conv', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)),
        #     ('toplayer2_BN', nn.BatchNorm2d(256)), ]))  # Reduce channels
        # self.toplayer1 = nn.Sequential(OrderedDict([
        #     ('toplayer1_conv', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)),
        #     ('toplayer1_BN', nn.BatchNorm2d(128)), ]))  # Reduce channels
        # self.toplayer0 = nn.Sequential(OrderedDict([
        #     ('toplayer0_conv', nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)),
        #     ('toplayer0_BN', nn.BatchNorm2d(32)), ]))  # Reduce channels

        self.loc_0 = nn.Sequential(OrderedDict([
            ('loc_0_conv', nn.Conv2d(128, 6 * 6, 3, padding=1)),
            ('loc_0_BN', nn.BatchNorm2d(36)), ]))
        self.loc_1 = nn.Sequential(OrderedDict([
            ('loc_1_conv', nn.Conv2d(256, 6 * 6, 3, padding=1)),
            ('loc_1_BN', nn.BatchNorm2d(36)), ]))
        self.loc_2 = nn.Sequential(OrderedDict([
            ('loc_2_conv', nn.Conv2d(512, 6 * 6, 3, padding=1)),
            ('loc_2_BN', nn.BatchNorm2d(36)), ]))
        self.loc_3 = nn.Sequential(OrderedDict([
            ('loc_3_conv', nn.Conv2d(1024, 6 * 6, 3, padding=1)),
            ('loc_3_BN', nn.BatchNorm2d(36)), ]))
        self.loc_4 = nn.Sequential(OrderedDict([
            ('loc_4_conv', nn.Conv2d(1024, 6 * 6, 3, padding=1)),
            ('loc_4_BN', nn.BatchNorm2d(36)), ]))
        self.loc_5 = nn.Sequential(OrderedDict([
            ('loc_5_conv', nn.Conv2d(512, 6 * 6, 3, padding=1)),
            ('loc_5_BN', nn.BatchNorm2d(36)), ]))
        self.loc_6 = nn.Sequential(OrderedDict([
            ('loc_6_conv', nn.Conv2d(512, 4 * 6, 3, padding=1)),
            ('loc_6_BN', nn.BatchNorm2d(24)), ]))

        self.conf_0 = nn.Sequential(OrderedDict([
            ('conf_0_conv', nn.Conv2d(128, 6 * self.num_classes, 3, padding=1)),
            ('conf_0_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))
        self.conf_1 = nn.Sequential(OrderedDict([
            ('conf_1_conv', nn.Conv2d(256, 6 * self.num_classes, 3, padding=1)),
            ('conf_1_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))

        self.conf_2 = nn.Sequential(OrderedDict([
            ('conf_2_conv', nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)),
            ('conf_2_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))
        self.conf_3 = nn.Sequential(OrderedDict([
            ('conf_3_conv', nn.Conv2d(1024, 6 * self.num_classes, 3, padding=1)),
            ('conf_3_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))
        self.conf_4 = nn.Sequential(OrderedDict([
            ('conf_4_conv', nn.Conv2d(1024, 6 * self.num_classes, 3, padding=1)),
            ('conf_4_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))
        self.conf_5 = nn.Sequential(OrderedDict([
            ('conf_5_conv', nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)),
            ('conf_5_BN', nn.BatchNorm2d(6 * self.num_classes)),
        ]))
        self.conf_6 = nn.Sequential(OrderedDict([
            ('conf_6_conv', nn.Conv2d(512, 4 * self.num_classes, 3, padding=1)),
            ('conf_6_BN', nn.BatchNorm2d(4 * self.num_classes)),
        ]))

        self.priorbox = PriorBox(solver)
        self.priors = Variable(self.priorbox.forward(), volatile=True)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        count_layer = 0
        f_2s = s0  # [N,32,h/2,w/2]
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1,0)
            count_layer += 1
            if count_layer ==6:  # [N,32*4,h/4,w/4]
                f0 = s1
            if count_layer == 9:  # [N,64*4,h/8,w/8]
                f1 = s1
            if count_layer == 12:  # [N,128*4,h/16,w/16]
                f2 = s1
            if count_layer == 13:
                det_s0 = seg_s0 = s1
            if count_layer == 14:  # [N,256*4,h/32,w/32]
                det_s1 = seg_s1 = feature3 = s1
        # pdb.set_trace()
        # count_layer = 0

        for i, cell in enumerate(self.detcells):
            det_s0, det_s1 = det_s1, cell(det_s0, det_s1,0)
            if i == 0:  # [N,32*4,h/4,w/4]
                feature4 = det_s1
            if i == 1:  # [N,64*4,h/8,w/8]
                det_s0 = feature5 = det_s1
            if i == 2:  # [N,128*4,h/16,w/16]
                feature6 = det_s1
        
        seg_s0 = seg_s1
        for i, cell in enumerate(self.segcells):
            # pdb.set_trace()
            seg_s0, seg_s1 = seg_s1, cell(seg_s0, seg_s1,0)
            # count_layer += 1
            if i ==0:  # [N,32*4,h/4,w/4]
                top3 = nn.functional.upsample(seg_s1, scale_factor=2)
                feature2 = seg_s0 = seg_s1 = top3 + f2  # 1024->512//
            if i ==1:  # [N,64*4,h/8,w/8]
                top2 = nn.functional.upsample(seg_s1, scale_factor=2)
                feature1 = seg_s0 = seg_s1  = top2 + f1  
            if i ==2:  # [N,128*4,h/16,w/16]
                top1 = nn.functional.upsample(seg_s1, scale_factor=2)
                feature0 = seg_s0 = seg_s1  = top1 + f0
            if i ==3:
                # pdb.set_trace()
                seg_feature = nn.functional.upsample(seg_s1, scale_factor=2) + f_2s
            
    
        # top3 = nn.functional.upsample(self.toplayer3(feature3), scale_factor=2)
        # feature2 = top3 + f2  # 1024->512//
        # top2 = nn.functional.upsample(self.toplayer2(feature2), scale_factor=2)
        # feature1 = top2 + f1  # 512->256//
        # top1 = nn.functional.upsample(self.toplayer1(feature1), scale_factor=2)
        # feature0 = top1 + f0  # 256->128//
        # seg_feature = nn.functional.upsample(self.toplayer0(feature0), scale_factor=2) + f_2s  # 128->64//
        logits_2s = self.score_2s(seg_feature)  # 64
        
        seg = nn.functional.upsample(logits_2s, scale_factor=2)
        
        
    
        # feature4 = self.conv_block6(feature3)
        # feature5 = self.conv_block7(feature4)
        # feature6 = self.conv_block8(feature5)
        # pdb.set_trace()
        loc = list()
        loc.append(self.loc_0(feature0).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_1(feature1).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_2(feature2).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_3(feature3).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_4(feature4).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_5(feature5).permute(0, 2, 3, 1).contiguous())
        loc.append(self.loc_6(feature6).permute(0, 2, 3, 1).contiguous())

        conf = list()
        conf.append(self.conf_0(feature0).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_1(feature1).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_2(feature2).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_3(feature3).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_4(feature4).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_5(feature5).permute(0, 2, 3, 1).contiguous())
        conf.append(self.conf_6(feature6).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        output = (loc.view(loc.size(0), -1, 6),
                  conf.view(conf.size(0), -1, self.num_classes),
                  seg,
                  self.priors)

        return output

    def drop_path(x, drop_prob):
        if drop_prob > 0.:
            keep_prob = 1. - drop_prob
            mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x.div_(keep_prob)
            x.mul_(mask)
        return x


def build_model(c, layers, genotype,detgenotype ,seggenotype , det_classes, seg_classes):
    model = Network(c, layers, genotype,detgenotype,seggenotype,det_classes, seg_classes)
    return model

