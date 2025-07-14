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
from data.TrainDataset import TrainDataset

torch.manual_seed(990919)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

task_name = 'Train-CharbonnierL1'
data_root = ''
checkpoint_path = os.path.join('checkpoints/', task_name)
'''
Parameters
'''
num_epochs = 200
batch_size = 4
learning_rate = 3e-6

model = Model()
model.load_model('weights/GMFSS', -1)
model.train()
model.device()
charbonnier_L1 = Charbonnier_L1().to(device)
optimizer = optim.AdamW(itertools.chain(
    model.metricnet.parameters(),
    model.feat_ext.parameters(),
    model.fusionnet.parameters(),
), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

dataloader = DataLoader(TrainDataset(root=data_root), batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)

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

    for frame0, frame1, frame2 in dataloader:
        frame0 = frame0.to(device)
        gt = frame1.to(device)
        frame1 = frame2.to(device)
        n, c, h, w = frame0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        frame0 = F.interpolate(frame0, (ph, pw), mode='bilinear', align_corners=False)
        frame1 = F.interpolate(frame1, (ph, pw), mode='bilinear', align_corners=False)

        out = model(frame0 / 255., frame1 / 255., timestep=0.5)
        out = out * 255.
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=False)
        loss = charbonnier_L1(out, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses += loss.item()
    end = time.time()
    avg_train_loss = train_losses / len(dataloader)
    scheduler.step(avg_train_loss)

    print(f"Epoch {epoch:>4} / {num_epochs} Train Loss: {avg_train_loss:<8.4f} Time: {(end - start) / 60:.2f} min")

    with open(f'{task_name}.log', 'a') as f:
        f.write(f'Epoch {epoch:>4} / {num_epochs} | Train Loss: {avg_train_loss:<8.4f} | Time: {(end - start) / 60:.2f} min\n')

    checkpoints = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    os.makedirs(os.path.join(checkpoint_path, f'{epoch}'))
    torch.save(checkpoints, os.path.join(checkpoint_path, f'{epoch}', 'misc.pkl'))
    model.save_model(os.path.join(checkpoint_path, f'{epoch}'), -1)

