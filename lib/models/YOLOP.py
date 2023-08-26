import torch
from torch import tensor
import torch.nn as nn
from torch.nn import Conv2d
import sys,os
import math
import sys
sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.common import GhostConv, RepConv, PaFPNELAN_C2, Conv, seg_head, PSA_p
from lib.models.common import ELANBlock_Head, FPN_C5, FPN_C2, ELANBlock_Head_Ghost, Repconv_Block, ELANNet, PaFPNELAN_Ghost_C2, IDetect

YOLOP = [
[3, 17, 27],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx

###### Backbone
[ -1, ELANNet, [True]],   #0

###### PaFPNELAN
[ -1, PaFPNELAN_Ghost_C2, []],   #1

###### Repconv_Block
[ 1, Repconv_Block, []],   #2

###### Detect Head
[ -1, IDetect,  [1, [[4.15629,  11.41984,  5.94761,  16.46950,  8.18673,  23.52688], [12.04416,  29.51737, 16.35089,  41.95507, 24.17928,  57.18741], [33.29597,  78.16243, 47.86408, 108.28889, 36.33312, 189.21414], [73.09806, 144.64581, 101.18080, 253.37000, 136.02821, 408.82248]], [128, 256, 512, 1024]]], #3 Detection head

######
[ 1, FPN_C2, []],   #4
[ 1, FPN_C5, []],   #5

###### drivable area segmentaiton head
[ 5, Conv, [512, 256, 3, 1]],  
[ -1, Upsample, [None, 2, 'bilinear', True]],  
[ -1, ELANBlock_Head, [256, 128]],
[ -1, Conv, [128, 64, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear', True]], 
[ -1, Conv, [64, 32, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear', True]], 
[ -1, Conv, [32, 16, 3, 1]], 
[ -1, ELANBlock_Head, [16, 8]], 
[ -1, Upsample, [None, 2, 'bilinear', True]], 
[ -1, Conv, [8, 2, 3, 1]], 
[ -1, seg_head, ['sigmoid']],  #17

###### lane detection head
[ 4, ELANBlock_Head, [128, 64]], 
[ -1, PSA_p, [64, 64]],  
[ -1, Conv, [64, 32, 3, 1]], 
[ -1, Upsample, [None, 2, 'bilinear', True]], 
[ -1, Conv, [32, 16, 3, 1]], 
[ -1, ELANBlock_Head, [16, 8]],
[ -1, PSA_p, [8, 8]],   
[ -1, Upsample, [None, 2, 'bilinear', True]], 
[ -1, Conv, [8, 2, 3, 1]], 
[ -1, seg_head, ['sigmoid']],  #27

]

class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        # 27 
        self.det_out_idx = block_cfg[0][0]

        # 63 67
        self.seg_out_idx = block_cfg[0][1:]
    
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is IDetect:
                # detector_index  # 27
                self.detector_index = i

            block_ = block(*args)
    
            block_.index, block_.from_ = i, from_

            layers.append(block_)

            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, IDetect):
            s = 512  # 2x min stride
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects = model_out[0]
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward

            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                out.append(x)
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        for m in self.model.modules():
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, IDetect):
                m.fuse()
                m.forward = m.fuseforward
        return self            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

if __name__ == "__main__":
    pass
    # from torch.utils.tensorboard import SummaryWriter
    # model = get_net(False)
    # input_ = torch.randn((1, 3, 256, 256))
    # gt_ = torch.rand((1, 2, 256, 256))
    # metric = SegmentationMetric(2)
    # model_out,SAD_out = model(input_)
    # detects, dring_area_seg, lane_line_seg = model_out
    # Da_fmap, LL_fmap = SAD_out
    # for det in detects:
    #     print(det.shape)
    # print(dring_area_seg.shape)
    # print(lane_line_seg.shape)
 
