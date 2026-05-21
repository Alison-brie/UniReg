import torch
import torch.nn as nn
import torch.nn.functional as nnf
import inspect
import functools

# Python 3.11 removed inspect.getargspec; keep legacy decorator working.
if not hasattr(inspect, "getargspec"):
    from collections import namedtuple
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    def _compat_getargspec(func):
        fs = inspect.getfullargspec(func)
        return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
    inspect.getargspec = _compat_getargspec

import numpy as np
from torch.distributions.normal import Normal
import math

# from . import layers

def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    attrs, varargs, varkw, defaults = inspect.getargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper

class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model

class SpatialTransformer(nn.Module):
    """Voxel-space spatial transformer (original convention)."""

    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.unsqueeze(torch.stack(grids), 0).float()
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1, norm=None):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        
        self.norm = None
        if norm == 'instance':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm3d(out_channels)
            
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        if self.norm is not None:
             out = self.norm(out)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1, norm=None):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        
        self.norm = None
        if norm == 'instance':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm == 'batch':
            self.norm = nn.BatchNorm3d(out_channels)
            
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        if self.norm is not None:
             out = self.norm(out)
        out = x + out
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=8, norm=None):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c, norm=norm)
        self.block2 = ConvBlock(c, c * 2, norm=norm)
        self.block3 = ConvBlock(c *2, c * 4, norm=norm)
        self.block4 = ConvBlock(c *4, c * 4, norm=norm)

    def forward(self, x):
        out1 = self.block1(x)
        x = nn.AvgPool3d(2)(out1)
        out2 = self.block2(x)
        x = nn.AvgPool3d(2)(out2)
        out3 = self.block3(x)
        x = nn.AvgPool3d(2)(out3)
        out4 = self.block4(x)
        return out1, out2, out3, out4

class DecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel, norm=None):
        super(DecoderBlock, self).__init__()
        self.Conv1 = ConvBlock(x_channel+y_channel, out_channel, norm=norm)
        self.Conv2 = ConvResBlock(out_channel, out_channel, norm=norm)
        self.Conv3 = ConvResBlock(out_channel, out_channel, norm=norm)
        # self.Conv3_2 = ConvResBlock(out_channel, out_channel)
        # self.Conv3_3 = ConvResBlock(out_channel, out_channel)
        self.Conv4 = nn.Conv3d(out_channel, out_channel//2, 3, padding=1)
        self.Conv5 = nn.Conv3d(out_channel//2, 3, 3, padding=1)
        
        nn.init.zeros_(self.Conv5.weight)
        if self.Conv5.bias is not None:
            nn.init.zeros_(self.Conv5.bias)

    def forward(self, x, y):
        try:
            concat = torch.cat([x, y], dim=1)
        except RuntimeError as e:
            # 关键维度保护：如果再次出现坏数据，给出清晰报错而不是含糊的 Dimension Error
            print(f"[ERROR] DecoderBlock concat failed: x={x.shape}, y={y.shape}. Input data mismatch?")
            raise e
            
        cost_vol = self.Conv1(concat)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        # cost_vol = self.Conv3_2(cost_vol)
        # cost_vol = self.Conv3_3(cost_vol)
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow

class RPNet(LoadableModel):
    @store_config_args
    def __init__(self, size, in_channel=1, first_channel=8, shared_encoder=True, norm=None):
        super(RPNet, self).__init__()
        self.shared_encoder = shared_encoder
        c = first_channel
        self.encoder = Encoder(in_channel, c, norm=norm)
        
        # [NEW] Optional Fixed Encoder for Cross-Modal (CT-MRI)
        if not self.shared_encoder:
            print("🚀 [Model] Using Independent Encoders for Moving/Fixed images (Cross-Modal Mode)")
            self.encoder_fixed = Encoder(in_channel, c, norm=norm)
        else:
            self.encoder_fixed = None
            
        # self.decoder4 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32, norm=norm)
        # self.decoder3 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32, norm=norm)
        # self.decoder2 = DecoderBlock(x_channel = 16, y_channel = 16, out_channel = 32, norm=norm)
        # self.decoder1 = DecoderBlock(x_channel = 8, y_channel = 8, out_channel = 16, norm=norm)
        
        self.decoder4 = DecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, norm=norm)
        self.decoder3 = DecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, norm=norm)
        self.decoder2 = DecoderBlock(x_channel=2*c, y_channel=2*c, out_channel=4*c, norm=norm)
        self.decoder1 = DecoderBlock(x_channel=c,   y_channel=c,   out_channel=2*c, norm=norm)
        
        self.size = size

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def _warp_feat(self, feat, flow, level):
        """Warp feat by flow at given pyramid level."""
        return self.transformer[level](feat, flow)

    def forward(self, x, y):
        fx1, fx2, fx3, fx4 = self.encoder(x)
        if self.shared_encoder:
            fy1, fy2, fy3, fy4 = self.encoder(y)
        else:
            fy1, fy2, fy3, fy4 = self.encoder_fixed(y)

        ar = br = cr = dr = 1
        wx4 = fx4

        flowall = None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for bb in range(br):
            wx3 = self.transformer[2](fx3, flowall)
            flow = self.decoder3(wx3, fy3)
            flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for cc in range(cr):
            wx2 = self.transformer[1](fx2, flowall)
            flow = self.decoder2(wx2, fy2)
            flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for dd in range(dr):
            wx1 = self.transformer[0](fx1, flowall)
            flow = self.decoder1(wx1, fy1)
            flowall = self.transformer[0](flowall, flow) + flow
        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall

class IIRPNet(LoadableModel):
    @store_config_args
    def __init__(self, size, in_channel=1, first_channel=8, shared_encoder=True, norm=None):
        super(IIRPNet, self).__init__()
        self.shared_encoder = shared_encoder
        c = first_channel
        self.encoder = Encoder(in_channel, c, norm=norm)
        
        # [NEW] Independent Encoder
        if not self.shared_encoder:
            print("🚀 [IIRPNet] Using Independent Encoders for Moving/Fixed images")
            self.encoder_fixed = Encoder(in_channel, c, norm=norm)
        else:
            self.encoder_fixed = None
            
        # self.decoder4 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32, norm=norm)
        # self.decoder3 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32, norm=norm)
        # self.decoder2 = DecoderBlock(x_channel = 16, y_channel = 16, out_channel = 32, norm=norm)
        # self.decoder1 = DecoderBlock(x_channel = 8, y_channel = 8, out_channel = 16, norm=norm)
        
        self.decoder4 = DecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, norm=norm)
        self.decoder3 = DecoderBlock(x_channel=4*c, y_channel=4*c, out_channel=4*c, norm=norm)
        self.decoder2 = DecoderBlock(x_channel=2*c, y_channel=2*c, out_channel=4*c, norm=norm)
        self.decoder1 = DecoderBlock(x_channel=c,   y_channel=c,   out_channel=2*c, norm=norm)
        
        self.size = size

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def _warp_feat(self, feat, flow, level):
        return self.transformer[level](feat, flow)

    def normalized_cross_correlation(self, img1, img2):
        # 计算均值
        mean_img1 = torch.mean(img1)
        mean_img2 = torch.mean(img2)

        # 计算标准差
        std_img1 = torch.std(img1)
        std_img2 = torch.std(img2)

        # 计算NCC
        ncc = torch.mean((img1 - mean_img1) * (img2 - mean_img2) / (std_img1 * std_img2))

        return ncc

    def pnsr(self, original_image, processed_image):
        # 确保输入张量的数据类型为浮点数
        original_image = original_image.float()
        processed_image = processed_image.float()

        # 计算均方误差（MSE）
        mse = torch.mean((original_image - processed_image) ** 2)

        # 如果MSE为0，PSNR无穷大，这里我们使用一个很小的非零值替代0以避免错误
        if mse == 0:
            return float('inf')

        # 计算PSNR
        max_intensity = 1.0  # 假设图像像素值的范围在0到255之间
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)

        return psnr

    def forward(self, x, y):
        fx1, fx2, fx3, fx4 = self.encoder(x)
        
        if self.shared_encoder:
            fy1, fy2, fy3, fy4 = self.encoder(y)
        else:
            fy1, fy2, fy3, fy4 = self.encoder_fixed(y)

        ar = 10
        br = 10
        cr = 10
        dr = 10

        pa = pb = pc = pd = 0

        current_iter = []
        wx4 = fx4
        mse_a = 100 #mse mae
        ncc_a = 0 #ncc,pnsr
        delta1 =0.005
        delta2 =0.005
        delta3 =0.005
        delta4 =0.005

        flowall = None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            previous_flow = flowall
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow) + flow
            wx4 = self.transformer[3](fx4, flowall)
            flowx4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8 * flowall)
            mx4 = self.transformer[0](x, flowx4)
            ncc = self.normalized_cross_correlation(mx4, y)
            ig = ncc < (ncc_a + delta1)
            if ig and previous_flow is not None:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pa += 1
        current_iter.append(pa)

        flowall = self.up(2 * flowall)
        mse_a = 100
        ncc_a = 0
        previous_flow = flowall
        for bb in range(br):
            previous_flow = flowall
            wx3 = self.transformer[2](fx3, flowall)
            flowx3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4 * flowall)
            mx3 = self.transformer[0](x, flowx3)
            ncc = self.normalized_cross_correlation(mx3, y)
            ig = ncc < (ncc_a + delta1)
            if ig and previous_flow is not None:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pb += 1

            flow = self.decoder3(wx3, fy3)
            previous_flow = flowall
            flowall = self.transformer[2](flowall, flow) + flow
        current_iter.append(pb)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for cc in range(cr):
            previous_flow = flowall
            wx2 = self.transformer[1](fx2, flowall)
            flowx2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2 * flowall)
            mx2 = self.transformer[0](x, flowx2)
            ncc = self.normalized_cross_correlation(mx2, y)
            ig = ncc < (ncc_a + delta1)
            if ig and previous_flow is not None:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pc += 1

            flow = self.decoder2(wx2, fy2)
            previous_flow = flowall
            flowall = self.transformer[1](flowall, flow) + flow
        current_iter.append(pc)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for dd in range(dr):
            previous_flow = flowall
            wx1 = self.transformer[0](fx1, flowall)
            mx = self.transformer[0](x, flowall)
            ncc = self.normalized_cross_correlation(mx, y)
            ig = ncc < (ncc_a + delta1)
            if ig and previous_flow is not None:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pd += 1

            flow = self.decoder1(wx1, fy1)
            previous_flow = flowall
            flowall = self.transformer[0](flowall, flow) + flow
        current_iter.append(pd)

        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall
