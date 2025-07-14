import os
import random
import torch
import cv2
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

class TrainDataset(Dataset):
    def __init__(self, root, crop_size=(288, 288), train=True):
        self.root = root
        self.train = train
        self.crop_size = crop_size
        self.frames = []

        for scene in os.listdir(self.root):
            frames = sorted(os.listdir(os.path.join(self.root, scene)))
            # Forms triplets
            for i in range(len(frames) - 2):
                triplet = (
                    os.path.join(self.root, scene, frames[i]),
                    os.path.join(self.root, scene, frames[i+1]),
                    os.path.join(self.root, scene, frames[i+2]),
                )
                self.frames.append(triplet)

    def transform(self, frames):
        ret = []
        i, j, h, w = v2.RandomCrop.get_params(frames[0], output_size=self.crop_size)
        horizontal_flip = random.random()
        vertical_flip = random.random()
        p = random.uniform(0, 1)

        for frame in frames:
            if self.train:
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

        # Reverse frames order
        if self.train:
            p = random.random()
            if p > 0.5:
                ret.reverse()
        return ret

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        paths = self.frames[index]
        frame0 = Image.open(paths[0]).convert('RGB')
        frame1 = Image.open(paths[1]).convert('RGB')
        frame2 = Image.open(paths[2]).convert('RGB')
        frames = self.transform((frame0, frame1, frame2))
        return frames[0], frames[1], frames[2]

if __name__ == "__main__":
    d = TrainDataset(root='/home/soda/Dataset/traindata')
    print(len(d))

    frame0, frame1, frame2, timestep = d[1]
    f0 = TF.to_pil_image(frame0)
    f1 = TF.to_pil_image(frame1)
    f2 = TF.to_pil_image(frame2)
    f0.save("f0.png")
    f1.save("f1.png")
    f2.save("f2.png")
