import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import itertools

from model.LineDiff import Model
from loss.Charbonnier_L1 import Charbonnier_L1
from loss.VGGPerceptualLoss import VGGPerceptualLoss
from data.AnimeRunFlow import AnimeRun

torch.manual_seed(990919)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

task_name = 'GMFSS-Flow-guided'
data_root = 'AnimeRun_v2/'
checkpoint_path = os.path.join('checkpoints/', task_name)
'''
Parameters
'''
num_epochs = 50
batch_size = 12
learning_rate = 9e-6

model = Model()
model.load_model('weights/GMFSS', -1)
model.train()
model.device()
criterion = nn.L1Loss()
# charbonnier = Charbonnier_L1().to(device)
# lpips = VGGPerceptualLoss(num_classes=1000, pretrained=False).to(device)
# lpips.load_weight('weights/sketch-FreezeConv3_4.pth')
optimizer = optim.AdamW(itertools.chain(
    model.metricnet.parameters(),
    model.feat_ext.parameters(),
    model.fusionnet.parameters(),
), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

dataloader = DataLoader(AnimeRun(root=data_root), batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
eval_dataloader = DataLoader(AnimeRun(root=data_root, train=False), batch_size=batch_size, num_workers=2)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# Check for the latest checkpoint
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoints = [int(f) for f in os.listdir(checkpoint_path)]
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        checkpoint_folder = os.path.join(checkpoint_path, f"{latest_checkpoint}")
        if os.path.exists(checkpoint_folder):
            checkpoint = torch.load(os.path.join(checkpoint_folder, 'misc.pkl'))
            model.load_model(checkpoint_folder, -1)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = latest_checkpoint
            print(f'Resuming from epoch {start_epoch}')

train_losses = []
for epoch in range(start_epoch + 1, num_epochs + 1):
    model.train()
    train_losses = 0.0
    start = time.time()

    for frame0, flow01, gt, flow10, frame1 in dataloader:
        frame0 = frame0.to(device)
        gt = gt.to(device)
        frame1 = frame1.to(device)
        flow01 = flow01.to(device)
        flow10 = flow10.to(device)

        out = model.forward2(frame0 / 255., frame1 / 255., flow01, flow10)
        out = out * 255.
        # out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        # loss = charbonnier(out - gt) + lpips(out, gt)
        loss = criterion(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
    end = time.time()
    avg_train_loss = train_losses / len(dataloader)
    scheduler.step(avg_train_loss)

    model.eval()
    eval_losses = 0.0
    for frame0, flow01, gt, flow10, frame1 in eval_dataloader:
        frame0 = frame0.to(device)
        gt = gt.to(device)
        frame1 = frame1.to(device)
        flow01 = flow01.to(device)
        flow10 = flow10.to(device)

        out = model.forward2(frame0 / 255., frame1 / 255., flow01, flow10)
        out = out * 255.
        loss = criterion(out, gt)
        eval_losses += loss.item()

    avg_eval_loss = eval_losses / len(eval_dataloader)
    print(f"Epoch {epoch:>4} / {num_epochs} Train Loss: {avg_train_loss:<8.4f} Eval Loss: {avg_eval_loss:<8.4f} Time: {(end - start) / 60:.2f} min")

    with open(f'{task_name}.log', 'a') as f:
        f.write(f"Epoch {epoch:>4} / {num_epochs} Train Loss: {avg_train_loss:<8.4f} Eval Loss: {avg_eval_loss:<8.4f} Time: {(end - start) / 60:.2f} min\n")

    checkpoints = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    os.makedirs(os.path.join(checkpoint_path, f'{epoch}'))
    torch.save(checkpoints, os.path.join(checkpoint_path, f'{epoch}', 'misc.pkl'))
    model.save_model(os.path.join(checkpoint_path, f'{epoch}'), -1)
