import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Ref. https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class StyleLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(StyleLoss, self).__init__()
        # Alphas
        self.weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval()) # relu1_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval()) # relu2_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:14].eval()) # relu3_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[18:23].eval()) # relu4_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[27:32].eval()) # relu5_2
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1))

    def vgg_loss(self, x, y):
        loss = 0.0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y) * self.weights[i]
        return loss

    def gram_loss(self, x, y):
        loss = 0.0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            y = block(y)
            act_x = x.reshape(x.shape[0], x.shape[1], -1)
            act_y = y.reshape(y.shape[0], y.shape[1], -1)
            gram_x = act_x @ act_x.permute(0, 2, 1)
            gram_y = act_y @ act_y.permute(0, 2, 1)
            loss += F.mse_loss(gram_x, gram_y) * self.weights[i]
        return loss

    def forward(self, output, gt):
        # Refer to FILM Section 4
        alphas = (1.0, 0.25, 40.0)

        if output.shape[1] != 3:
            output = output.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        output = (output-self.mean) / self.std
        gt = (gt-self.mean) / self.std
        if self.resize:
            output = F.interpolate(output, mode='bilinear', size=(224, 224), align_corners=False)
            gt = F.interpolate(gt, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = output
        y = gt

        return alphas[0] * F.l1_loss(output, gt) + alphas[1] * self.vgg_loss(x, y) + alphas[2] * self.gram_loss(x, y)
