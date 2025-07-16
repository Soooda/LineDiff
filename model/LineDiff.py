import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flownet.flowdiffuser import FlowDiffuser
from .flownet.utils.utils import InputPadder
from .MetricNet import MetricNet
from .FeatureNet import FeatureNet
from .FusionNet import GridNet
HAS_CUDA = True
try:
    import cupy
    if cupy.cuda.get_cuda_path() == None:
        HAS_CUDA = False
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from .warp.softsplat import softsplat as warp
else:
    print("System does not have CUPY installed, falling back to PyTorch")
    from .warp.softsplat_torch import softsplat as warp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        args = argparse.Namespace()
        args.model = None
        args.dataset = None
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False
        self.flownet = nn.DataParallel(FlowDiffuser(args))
        # self.flownet.load_state_dict(torch.load('weights/FlowDiffuser-things.pth'))
        self.flownet.load_state_dict(torch.load('weights/10000_fd-animerun.pth'))
        self.metricnet = MetricNet()
        self.feat_ext = FeatureNet()
        self.fusionnet = GridNet()

    def train(self):
        self.flownet.eval()
        self.metricnet.train()
        self.feat_ext.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.metricnet.eval()
        self.feat_ext.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.metricnet.to(device)
        self.feat_ext.to(device)
        self.fusionnet.to(device)

    def save_model(self, path, rank=0):
        torch.save(self.metricnet.state_dict(), f'{path}/metric.pkl')
        torch.save(self.feat_ext.state_dict(), f'{path}/feat.pkl')
        torch.save(self.fusionnet.state_dict(), f'{path}/fusionnet.pkl')

    def load_model(self, path, rank):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        # self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path),map_location=device))
        self.metricnet.load_state_dict(torch.load('{}/metric.pkl'.format(path),map_location=device))
        self.feat_ext.load_state_dict(torch.load('{}/feat.pkl'.format(path),map_location=device))
        self.fusionnet.load_state_dict(torch.load('{}/fusionnet.pkl'.format(path),map_location=device))

    def forward(self, img0, img1, timestep, scale=1.0):
        feat11, feat12, feat13 = self.feat_ext(img0)
        feat21, feat22, feat23 = self.feat_ext(img1)

        img0 = F.interpolate(img0, scale_factor = 0.5, mode="bilinear", align_corners=False)
        img1 = F.interpolate(img1, scale_factor = 0.5, mode="bilinear", align_corners=False)

        if scale != 1.0:
            imgf0 = F.interpolate(img0, scale_factor=scale, mode="bilinear", align_corners=False)
            imgf1 = F.interpolate(img1, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            imgf0 = img0
            imgf1 = img1

        padder = InputPadder(imgf0.shape)
        imgf0, imgf1 = padder.pad(imgf0, imgf1)

        flow_low, flow01 = self.flownet(imgf0, imgf1, iters=32, test_mode=True)
        flow01 = padder.unpad(flow01)
        flow_low, flow10 = self.flownet(imgf1, imgf0, iters=32, test_mode=True)
        flow10 = padder.unpad(flow10)

        metric0, metric1 = self.metricnet(img0, img1, flow01, flow10)

        F1t = timestep * flow01
        F2t = (1-timestep) * flow10

        Z1t = timestep * metric0
        Z2t = (1-timestep) * metric1

        I1t = warp(img0, F1t, Z1t, strMode='soft')
        I2t = warp(img1, F2t, Z2t, strMode='soft')

        feat1t1 = warp(feat11, F1t, Z1t, strMode='soft')
        feat2t1 = warp(feat21, F2t, Z2t, strMode='soft')

        F1td = F.interpolate(F1t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
        Z1d = F.interpolate(Z1t, scale_factor = 0.5, mode="bilinear", align_corners=False)
        feat1t2 = warp(feat12, F1td, Z1d, strMode='soft')
        F2td = F.interpolate(F2t, scale_factor = 0.5, mode="bilinear", align_corners=False) * 0.5
        Z2d = F.interpolate(Z2t, scale_factor = 0.5, mode="bilinear", align_corners=False)
        feat2t2 = warp(feat22, F2td, Z2d, strMode='soft')

        F1tdd = F.interpolate(F1t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
        Z1dd = F.interpolate(Z1t, scale_factor = 0.25, mode="bilinear", align_corners=False)
        feat1t3 = warp(feat13, F1tdd, Z1dd, strMode='soft')
        F2tdd = F.interpolate(F2t, scale_factor = 0.25, mode="bilinear", align_corners=False) * 0.25
        Z2dd = F.interpolate(Z2t, scale_factor = 0.25, mode="bilinear", align_corners=False)
        feat2t3 = warp(feat23, F2tdd, Z2dd, strMode='soft')

        out = self.fusionnet(torch.cat([img0, I1t, I2t, img1], dim=1), torch.cat([feat1t1, feat2t1], dim=1), torch.cat([feat1t2, feat2t2], dim=1), torch.cat([feat1t3, feat2t3], dim=1))

        return torch.clamp(out, 0, 1)
