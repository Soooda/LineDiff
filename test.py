import argparse
import torch
import torch.nn.functional as F
import cv2

from model.LineDiff import Model

parser = argparse.ArgumentParser()
parser.add_argument('--frame1', type=str, help='Frame 1 Path')
parser.add_argument('--frame2', type=str, help='Frame 2 Path')
parser.add_argument('--output', type=str, help='Output Path')
args = parser.parse_args()

path_to_img1 = args.frame1
path_to_img2 = args.frame2
path_to_weights = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(img, device=device):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

model = Model()
model.load_model(path_to_weights, -1)
model.eval()
model.device()

scale = 1.0
img0 = cv2.imread(path_to_img1)
img1 = cv2.imread(path_to_img2)
img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
n, c, h, w = img0.shape
tmp = max(64, int(64 / scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
img0 = F.interpolate(img0, (ph, pw), mode='bilinear', align_corners=False)
img1 = F.interpolate(img1, (ph, pw), mode='bilinear', align_corners=False)

out = model(img0, img1, timestep=0.5, scale=scale)
out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
out = (((out[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))[:h, :w])
cv2.imwrite(args.output, out)

