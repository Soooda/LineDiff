import os
import random
import torch
import cv2
from PIL import Image
import numpy as np
import sys

sys.path.append('model')

from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

from flownet.utils.frame_utils import read_gen
from flownet.utils.flow_viz import flow_to_image

class AnimeRun(Dataset):
    def __init__(self, root, crop_size=(416, 416), train=True):
        if train:
            self.root = os.path.join(root, 'train', 'contour')
            self.flow_root = os.path.join(root, 'train', 'Flow')
        else:
            self.root = os.path.join(root, 'test', 'contour')
            self.flow_root = os.path.join(root, 'test', 'Flow')
        self.train = train
        self.crop_size = crop_size
        self.frames = []

        for scene in os.listdir(self.root):
            frames = sorted(os.listdir(os.path.join(self.root, scene)))
            # Forms triplets
            for i in range(len(frames) - 2):
                data = (
                    scene,
                    os.path.join(self.root, scene, frames[i]),
                    os.path.join(self.root, scene, frames[i+1]),
                    os.path.join(self.root, scene, frames[i+2]),
                )
                self.frames.append(data)

    def transform(self, frames):
        scene = frames[0][0]
        ret = []
        i, j, h, w = v2.RandomCrop.get_params(frames[1], output_size=self.crop_size)
        horizontal_flip = random.random()
        vertical_flip = random.random()

        flow01 = read_gen(os.path.join(self.flow_root, scene, 'forward', frames[0][1].split('/')[-1].split('.')[0] + '.flo'))
        flow01 = np.array(flow01).astype(np.float32)
        flow01 = torch.from_numpy(flow01).permute(2, 0, 1).float()
        flow10 = read_gen(os.path.join(self.flow_root, scene, 'backward', frames[0][3].split('/')[-1].split('.')[0] + '.flo'))
        flow10 = np.array(flow10).astype(np.float32)
        flow10 = torch.from_numpy(flow10).permute(2, 0, 1).float()
        temp = [
            frames[1],
            flow01,
            frames[2],
            flow10,
            frames[3],
        ]

        p = random.uniform(0, 1)
        for frame in temp:
            # Random Crop
            frame = TF.crop(frame, i, j, h, w)
            # Random horizontal flipping
            if horizontal_flip > 0.5:
                frame = TF.hflip(frame)
            # Random vertical flipping
            if vertical_flip > 0.5:
                frame = TF.vflip(frame)
            # Random rotation
            if p < 0.25:
                frame = TF.rotate(frame, 90)
            elif p < 0.5:
                frame = TF.rotate(frame, 180)
            elif p < 0.75:
                frame = TF.rotate(frame, -90)
            frame = TF.to_dtype(TF.to_image(frame), dtype=torch.float32, scale=True)
            ret.append(frame)
        return ret

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        paths = self.frames[index]
        frame0 = Image.open(paths[1]).convert('RGB')
        gt = Image.open(paths[2]).convert('RGB')
        frame1 = Image.open(paths[3]).convert('RGB')
        if self.train:
            frames = self.transform((paths, frame0, gt, frame1))
        else:
            flow01 = read_gen(os.path.join(self.flow_root, paths[0], 'forward', paths[1].split('/')[-1].split('.')[0] + '.flo'))
            flow01 = np.array(flow01).astype(np.float32)
            flow01 = torch.from_numpy(flow01).permute(2, 0, 1).float()
            flow10 = read_gen(os.path.join(self.flow_root, paths[0], 'backward', paths[3].split('/')[-1].split('.')[0] + '.flo'))
            flow10 = np.array(flow10).astype(np.float32)
            flow10 = torch.from_numpy(flow10).permute(2, 0, 1).float()
            frames = [
                    TF.to_dtype(TF.to_image(frame0), dtype=torch.float32, scale=True),
                    flow01,
                    TF.to_dtype(TF.to_image(gt), dtype=torch.float32, scale=True),
                    flow10,
                    TF.to_dtype(TF.to_image(frame1), dtype=torch.float32, scale=True),
            ]
            frames = [TF.center_crop(f, self.crop_size) for f in frames]
        return frames[0], frames[1], frames[2], frames[3], frames[4]

if __name__ == "__main__":
    d = AnimeRun(root='/home/soda/Dataset/AnimeRun_v2')
    print(len(d))

    frame0, flow01, gt, flow10, frame1 = d[300]
    f0 = TF.to_pil_image(frame0)
    flow01 = flow01.permute(1, 2, 0).cpu().numpy()
    f1 = flow_to_image(flow01)
    cv2.imwrite('flow01.png', f1)
    flow10 = flow10.permute(1, 2, 0).cpu().numpy()
    f2 = flow_to_image(flow10)
    cv2.imwrite('flow10.png', f2)
    f3 = TF.to_pil_image(frame1)
    gt = TF.to_pil_image(gt)
    f0.save("f0.png")
    f3.save("f1.png")
    gt.save("gt.png")
