import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import torch.nn.functional as F
from torch.nn import Upsample
from torch.nn import SiLU
from torch.nn import Conv2d

class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    policy
                )
            )
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy)
            )


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DepthSeperabelConv2d(nn.Module):
    """
    DepthSeperable Convolution 2d with residual connection
    """

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, act=True):
        super(DepthSeperabelConv2d, self).__init__()
        # self.depthwise = nn.Sequential(
        #     nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=kernel_size//2, bias=False),
        #     nn.BatchNorm2d(inplanes, momentum=0.01, eps=1e-3)
        # )

        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride=stride, groups=inplanes, padding=1, bias=False)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, bias=False)

        # self.pointwise = nn.Sequential(
        #     nn.Conv2d(inplanes, planes, 1, bias=False),
        #     nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3)
        # )

        self.norm = nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3)
        self.downsample = downsample
        self.stride = stride
        try:
            self.act = SiLU() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        #residual = x

        out = self.depthwise(x)
        # out = self.act(out)
        out = self.pointwise(out)
        out = self.norm(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.act(out)

        return out

class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
        kenel_weight = np.vstack([sobel_kernel]*c2*c1).reshape(c2,c1,3,3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = SiLU() if act else nn.Identity()
        except:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0., 6.) / 6.  # for torchscript, CoreML and ONNX

# cbh
# [k=3, s=2] -> H/2,W/2
# [k=3, s=1] -> H,W
# [k=1, s=1] -> H,W
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# cbh
# [k=4, s=2, p=1] -> 2H,2W
class TransConv(nn.Module):
    # transpose convolution
    def __init__(self, c1, c2,  g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(TransConv, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, 4, 2, 1, groups=g, bias=False)
        self.isact=act
        if self.isact:
            self.act = SiLU()
            self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        if self.isact:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

# Res unit
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# yolo v5 csp1_x
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class C3Ghost_Backbone(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = Conv(c1, c_, 3, 2)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_, 3, 2) for _ in range(n)))

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class SPPF(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv5-SPPF
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size= k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)

class Detect(nn.Module):
    stride = None  # strides computed during build
    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes # 1
        self.no = nc + 5  # number of outputs per anchor # 1+5
        self.nl = len(anchors)  # number of detection layers # 5
        self.na = len(anchors[0]) // 2  # number of anchors # 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid 
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)   # shape(5,3,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)   # shape(5,1,3,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv   # Channel = [128, 256, 512, 768, 1024]

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # prediction conv

            bs, _, ny, nx = x[i].shape  

            x[i]=x[i].view(bs, self.na, self.no, ny*nx).permute(0, 1, 3, 2).view(bs, self.na, ny, nx, self.no).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no * self.na) for _ in ch)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)
    
    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1,c2,_,_ = self.m[i].weight.shape
            c1_,c2_, _,_ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1,c2),self.ia[i].implicit.reshape(c2_,c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix                          
        return (box, score)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super().__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.bn = nn.BatchNorm2d(self.inplanes, momentum=0.01, eps=1e-3)
        self.act = SiLU()

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1,2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):

        batch, channel, height, width = x.size()

        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel

        out = self.bn(out)
        out = self.act(out)

        return out

class ELANBlock(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act=True):
        super(ELANBlock, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv3 = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, act=act),
            Conv(inter_dim, inter_dim, k=3, act=act)
        )
        self.cv4 = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, act=act),
            Conv(inter_dim, inter_dim, k=3, act=act)
        )
 
        assert inter_dim*4 == out_dim
 
        self.out = Conv(inter_dim*4, out_dim, k=1)
 
    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
 
        # [B, C, H, W] -> [B, 2C, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4], dim=1))
 
        return out

class DownSample(nn.Module):
    """
    ELAN BLock of YOLOv7's backbone
    """
    def __init__(self, in_dim, act=True):
            super().__init__()
            inter_dim = in_dim // 2
            self.mp = nn.MaxPool2d(2, 2)
            self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
            self.cv2 = nn.Sequential(
                Conv(in_dim, inter_dim, k=1, act=act),
                Conv(inter_dim, inter_dim, k=3, s=2, act=act)
            )
    
    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)

        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)

        return out

class ELANBlock_Head(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act=True):
        super(ELANBlock_Head, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv3 = Conv(inter_dim, inter_dim2, k=3, act=act)
        self.cv4 = Conv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv5 = Conv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv6 = Conv(inter_dim2, inter_dim2, k=3, act=act)
 
        self.out = Conv(inter_dim*2+inter_dim2*4, out_dim, k=1, act=act)
 
    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
 
        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))

        return out
 
class DownSample_Head(nn.Module):
    def __init__(self, in_dim, act=True):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d(2, 2)
        self.cv1 = Conv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act=act),
            Conv(inter_dim, inter_dim, k=3, s=2, act=act)
        )
 
    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
 
        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)
 
        return out

class ELANBlock_Head_Ghost(nn.Module):
    """
    ELAN BLock of YOLOv7's head
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, act=True):
        super(ELANBlock_Head_Ghost, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        inter_dim2 = int(inter_dim * expand_ratio)
        self.cv1 = GhostConv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = GhostConv(in_dim, inter_dim, k=1, act=act)
        self.cv3 = GhostConv(inter_dim, inter_dim2, k=3, act=act)
        self.cv4 = GhostConv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv5 = GhostConv(inter_dim2, inter_dim2, k=3, act=act)
        self.cv6 = GhostConv(inter_dim2, inter_dim2, k=3, act=act)
 
        self.out = GhostConv(inter_dim*2+inter_dim2*4, out_dim, k=1, act=act)
 
 
    def forward(self, x):
        """
        Input:
            x: [B, C_in, H, W]
        Output:
            out: [B, C_out, H, W]
        """
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.cv3(x2)
        x4 = self.cv4(x3)
        x5 = self.cv5(x4)
        x6 = self.cv6(x5)
 
        # [B, C_in, H, W] -> [B, C_out, H, W]
        out = self.out(torch.cat([x1, x2, x3, x4, x5, x6], dim=1))
 
        return out
 
class DownSample_Head_Ghost(nn.Module):
    def __init__(self, in_dim, act=True):
        super().__init__()
        inter_dim = in_dim
        self.mp = nn.MaxPool2d(2, 2)
        self.cv1 = GhostConv(in_dim, inter_dim, k=1, act=act)
        self.cv2 = nn.Sequential(
            GhostConv(in_dim, inter_dim, k=1, act=act),
            GhostConv(inter_dim, inter_dim, k=3, s=2, act=act)
        )
 
    def forward(self, x):
        """
        Input:
            x: [B, C, H, W]
        Output:
            out: [B, 2C, H//2, W//2]
        """
        # [B, C, H, W] -> [B, C//2, H//2, W//2]
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
 
        # [B, C, H//2, W//2]
        out = torch.cat([x1, x2], dim=1)
 
        return out

# ELANNet of YOLOv7
class ELANNet(nn.Module):
    """
    ELAN-Net of YOLOv7.
    """
    def __init__(self, use_C2 = False):
        super(ELANNet, self).__init__()
 
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3),      
            Conv(32, 64, k=3, s=2),
            Conv(64, 64, k=3)                                                   # P1/2
        )
        self.layer_2 = nn.Sequential(   
            Conv(64, 128, k=3, s=2),             
            ELANBlock(in_dim=128, out_dim=256, expand_ratio=0.5)                     # P2/4
        )
        self.layer_3 = nn.Sequential(
            DownSample(in_dim=256),             
            ELANBlock(in_dim=256, out_dim=512, expand_ratio=0.5)                     # P3/8
        )
        self.layer_4 = nn.Sequential(
            DownSample(in_dim=512),             
            ELANBlock(in_dim=512, out_dim=1024, expand_ratio=0.5)                    # P4/16
        )
        self.layer_5 = nn.Sequential(
            DownSample(in_dim=1024),             
            ELANBlock(in_dim=1024, out_dim=1024, expand_ratio=0.25)                  # P5/32
        )

        self.use_C2 = use_C2
 
    def forward(self, x):
        x = self.layer_1(x)
        c2 = self.layer_2(x)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        if self.use_C2:
            return c2, c3, c4, c5
        else:
            return c3, c4, c5

# PaFPN-ELAN (YOLOv7's)
class PaFPNELAN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 1024],
                 out_dim=[256, 512, 1024], 
                 act=True):
        super(PaFPNELAN, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5//2, 256, k=1, act=act)
        self.cv2 = Conv(c4, 256, k=1, act=act)
        self.head_elan_1 = ELANBlock_Head(in_dim=512,
                                     out_dim=256,
                                     act=act)
        # P4 -> P3
        self.cv3 = Conv(256, 128, k=1, act=act)
        self.cv4 = Conv(c3, 128, k=1, act=act)
        self.head_elan_2 = ELANBlock_Head(in_dim=256,
                                     out_dim=128,  # 128
                                     act=act)
        # bottom up
        # P3 -> P4
        self.mp1 = DownSample_Head(128, act=act)
        self.head_elan_3 = ELANBlock_Head(in_dim=512,
                                     out_dim=256,  # 256
                                     act=act)
        # P4 -> P5
        self.mp2 = DownSample_Head(256, act=act)
        self.head_elan_4 = ELANBlock_Head(in_dim=1024,
                                     out_dim=512,  # 512
                                     act=act)

        self.SPPCSPC = SPP(c5, 512)

    def forward(self, features):
        # c3, c4, c5
        _, c3, c4, c5 = features

        # SPP Module
        c5 = self.SPPCSPC(c5)

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)

        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        return c8, c12, c13, c16, c19

# PaFPN-ELAN (YOLOv7's)
class PaFPNELAN_Ghost(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 1024],
                 out_dim=[256, 512, 1024], 
                 act=True):
        super(PaFPNELAN_Ghost, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c3, c4, c5 = in_dims
        # top dwon
        ## P5 -> P4
        self.cv1 = GhostConv(c5//2, 256, k=1, act=act)
        self.cv2 = GhostConv(c4, 256, k=1, act=act)
        self.head_elan_1 = ELANBlock_Head_Ghost(in_dim=512,
                                     out_dim=256,
                                     act=act)
        # P4 -> P3
        self.cv3 = GhostConv(256, 128, k=1, act=act)
        self.cv4 = GhostConv(c3, 128, k=1, act=act)
        self.head_elan_2 = ELANBlock_Head_Ghost(in_dim=256,
                                     out_dim=128,  # 128
                                     act=act)
        # bottom up
        # P3 -> P4
        self.mp1 = DownSample_Head_Ghost(128, act=act)
        self.head_elan_3 = ELANBlock_Head_Ghost(in_dim=512,
                                     out_dim=256,  # 256
                                     act=act)
        # P4 -> P5
        self.mp2 = DownSample_Head_Ghost(256, act=act)
        self.head_elan_4 = ELANBlock_Head_Ghost(in_dim=1024,
                                     out_dim=512,  # 512
                                     act=act)

        self.SPPCSPC = GhostSPPCSPC(c5, 512)

    def forward(self, features):
        # c3, c4, c5
        c2, c3, c4, c5 = features

        # SPP Module
        c5 = self.SPPCSPC(c5)

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.mp1(c13)
        c15 = torch.cat([c14, c9], dim=1)
        c16 = self.head_elan_3(c15)

        # P4 -> P5
        c17 = self.mp2(c16)
        c18 = torch.cat([c17, c5], dim=1)
        c19 = self.head_elan_4(c18)

        return c2, c13, c16, c19

# PaFPN-ELAN (YOLOv7's)
class PaFPNELAN_C2(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024, 1024],
                 out_dim=[128, 256, 512, 1024], 
                 act=True):
        super(PaFPNELAN_C2, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c2, c3, c4, c5 = in_dims
        # top dwon
        ## P5 -> P4
        self.cv1 = Conv(c5//2, 256, k=1, act=act)
        self.cv2 = Conv(c4, 256, k=1, act=act)
        self.head_elan_1 = ELANBlock_Head(in_dim=512,
                                     out_dim=256,
                                     act=act)
        # P4 -> P3
        self.cv3 = Conv(256, 128, k=1, act=act)
        self.cv4 = Conv(c3, 128, k=1, act=act)
        self.head_elan_2 = ELANBlock_Head(in_dim=256,
                                     out_dim=128,  # 128
                                     act=act)

        # P3 -> P2
        self.cv5 = Conv(128, 64, k=1, act=act)
        self.cv6 = Conv(c2, 64, k=1, act=act)
        self.head_elan_3 = ELANBlock_Head(in_dim=128,
                                     out_dim=64,  # 128
                                     act=act)

        # bottom up
        # P2 -> P3
        self.mp0 = DownSample_Head(64, act=act)
        self.head_elan_4 = ELANBlock_Head(in_dim=256,
                                     out_dim=128,  # 256
                                     act=act)

        # P3 -> P4
        self.mp1 = DownSample_Head(128, act=act)
        self.head_elan_5 = ELANBlock_Head(in_dim=512,
                                     out_dim=256,  # 256
                                     act=act)
        # P4 -> P5
        self.mp2 = DownSample_Head(256, act=act)
        self.head_elan_6 = ELANBlock_Head(in_dim=1024,
                                     out_dim=512,  # 512
                                     act=act)

        self.SPPCSPC = SPPCSPC(c5, 512)

    def forward(self, features):
        # c3, c4, c5
        c2, c3, c4, c5 = features

        # SPP Module
        c5 = self.SPPCSPC(c5)

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        ## P3 -> P2
        c14 = self.cv5(c13)
        c15 = F.interpolate(c14, scale_factor=2.0)
        c16 = torch.cat([c15, self.cv6(c2)], dim=1)
        c17 = self.head_elan_3(c16)

        # Bottom up
        # p2 -> P3
        c18 = self.mp0(c17)
        c19 = torch.cat([c18, c13], dim=1)
        c20 = self.head_elan_4(c19)

        # P3 -> P4
        c21 = self.mp1(c20)
        c22 = torch.cat([c21, c9], dim=1)
        c23 = self.head_elan_5(c22)

        # P4 -> P5
        c24 = self.mp2(c23)
        c25 = torch.cat([c24, c5], dim=1)
        c26 = self.head_elan_6(c25)

        return c8, c16, c17, c20, c23, c26

# PaFPN-ELAN_Ghost (YOLOv7's)
class PaFPNELAN_Ghost_C2(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024, 1024],
                 out_dim=[128, 256, 512, 1024], 
                 act=True):
        super(PaFPNELAN_Ghost_C2, self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        c2, c3, c4, c5 = in_dims
        # top dwon
        ## P5 -> P4
        self.cv1 = GhostConv(c5//2, 256, k=1, act=act)
        self.cv2 = GhostConv(c4, 256, k=1, act=act)
        self.head_elan_1 = ELANBlock_Head_Ghost(in_dim=512,
                                     out_dim=256,
                                     act=act)
        # P4 -> P3
        self.cv3 = GhostConv(256, 128, k=1, act=act)
        self.cv4 = GhostConv(c3, 128, k=1, act=act)
        self.head_elan_2 = ELANBlock_Head_Ghost(in_dim=256,
                                     out_dim=128,  # 128
                                     act=act)

        # P3 -> P2
        self.cv5 = GhostConv(128, 64, k=1, act=act)
        self.cv6 = GhostConv(c2, 64, k=1, act=act)
        self.head_elan_3 = ELANBlock_Head_Ghost(in_dim=128,
                                     out_dim=64,  # 128
                                     act=act)

        # bottom up
        # P2 -> P3
        self.mp0 = DownSample_Head_Ghost(64, act=act)
        self.head_elan_4 = ELANBlock_Head_Ghost(in_dim=256,
                                     out_dim=128,  # 256
                                     act=act)

        # P3 -> P4
        self.mp1 = DownSample_Head_Ghost(128, act=act)
        self.head_elan_5 = ELANBlock_Head_Ghost(in_dim=512,
                                     out_dim=256,  # 256
                                     act=act)
        # P4 -> P5
        self.mp2 = DownSample_Head_Ghost(256, act=act)
        self.head_elan_6 = ELANBlock_Head_Ghost(in_dim=1024,
                                     out_dim=512,  # 512
                                     act=act)

        self.SPPCSPC = GhostSPPCSPC(c5, 512)

    def forward(self, features):
        # c3, c4, c5
        c2, c3, c4, c5 = features

        # SPP Module
        c5 = self.SPPCSPC(c5)

        # Top down
        ## P5 -> P4
        c6 = self.cv1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c8 = torch.cat([c7, self.cv2(c4)], dim=1)
        c9 = self.head_elan_1(c8)

        ## P4 -> P3
        c10 = self.cv3(c9)
        c11 = F.interpolate(c10, scale_factor=2.0)
        c12 = torch.cat([c11, self.cv4(c3)], dim=1)
        c13 = self.head_elan_2(c12)

        ## P3 -> P2
        c14 = self.cv5(c13)
        c15 = F.interpolate(c14, scale_factor=2.0)
        c16 = torch.cat([c15, self.cv6(c2)], dim=1)
        c17 = self.head_elan_3(c16)

        # Bottom up
        # p2 -> P3
        c18 = self.mp0(c17)
        c19 = torch.cat([c18, c13], dim=1)
        c20 = self.head_elan_4(c19)

        # P3 -> P4
        c21 = self.mp1(c20)
        c22 = torch.cat([c21, c9], dim=1)
        c23 = self.head_elan_5(c22)

        # P4 -> P5
        c24 = self.mp2(c23)
        c25 = torch.cat([c24, c5], dim=1)
        c26 = self.head_elan_6(c25)

        return c8, c16, c17, c20, c23, c26

class Repconv_Block(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, out_dim=[128, 256, 512, 1024] ):
        super(Repconv_Block, self).__init__()
        # RepConv
        self.repconv_0 = RepConv(64, out_dim[0], k=3, s=1, p=1)
        self.repconv_1 = RepConv(128, out_dim[1], k=3, s=1, p=1)
        self.repconv_2 = RepConv(256, out_dim[2], k=3, s=1, p=1)
        self.repconv_3 = RepConv(512, out_dim[3], k=3, s=1, p=1)

    def forward(self, x):
        _, _, c2,c3,c4,c5 = x
        # RepCpnv
        c27 = self.repconv_0(c2) # P2
        c28 = self.repconv_1(c3) # P3
        c29 = self.repconv_2(c4) # P4
        c30 = self.repconv_3(c5) # P5
        out_feats = [c27, c28, c29, c30] # [P2, P3, P4, P5]
        return out_feats

class Repconv_Block_NoC2(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, out_dim=[128, 256, 512, 1024] ):
        super(Repconv_Block_NoC2, self).__init__()
        # RepConv
        self.repconv_1 = RepConv(128, out_dim[1], k=3, s=1, p=1)
        self.repconv_2 = RepConv(256, out_dim[2], k=3, s=1, p=1)
        self.repconv_3 = RepConv(512, out_dim[3], k=3, s=1, p=1)

    def forward(self, x):
        _, _, c3,c4,c5 = x
        # RepCpnv
        c28 = self.repconv_1(c3) # P3
        c29 = self.repconv_2(c4) # P4
        c30 = self.repconv_3(c5) # P5
        out_feats = [c28, c29, c30] # [P2, P3, P4, P5]

        return out_feats

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)

##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
    
##### end of yolor #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        

        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None

class FPN_C2(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        _, n2 , _ , _ , _ , _ = x
        return n2

class FPN_C3(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        _, n2 , _ , _ , _ = x
        return n2

class FPN_C5_noC2(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        n3, _ , _ , _ , _  = x
        return n3

class FPN_C5(nn.Module):
    def __init__(self, dim = 256):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        n3, _ , _ , _ , _ , _ = x
        return n3

class seg_head(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, mode = 'sigmoid'):
        super(seg_head, self).__init__()
        # self.upsample_1 = nn.Upsample(size, scale_factor, mode, align_corners )
        # self.upsample_1 = nn.Upsample(size, scale_factor, mode, align_corners )
        if mode == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act =nn.Softmax(dim = 1)

    def forward(self, x):
        # x = self.upsample_1(x)
        x=x.float()

        if not self.training:  # inference
            x = self.act(x)
            
        return x