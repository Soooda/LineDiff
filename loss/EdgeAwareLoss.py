import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeAwareLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EdgeAwareLoss, self).__init__()
        self.reduction = reduction

        # Define Sobel kernels
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()

        self.sobel_x = sobel_x.view(1, 1, 3, 3)
        self.sobel_y = sobel_y.view(1, 1, 3, 3)

    def gradient_magnitude(self, img):
        # img: (B, C, H, W)
        B, C, H, W = img.size()
        device = img.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        grads_x = F.conv2d(img, sobel_x.expand(C, 1, 3, 3), groups=C, padding=1)
        grads_y = F.conv2d(img, sobel_y.expand(C, 1, 3, 3), groups=C, padding=1)

        grad_mag = torch.sqrt(grads_x ** 2 + grads_y ** 2 + 1e-6)
        return grad_mag

    def forward(self, pred, target):
        # pred, target: (B, 3, H, W), RGB images
        pred_grad = self.gradient_magnitude(pred)
        target_grad = self.gradient_magnitude(target)
        loss = F.l1_loss(pred_grad, target_grad, reduction=self.reduction)
        return loss
