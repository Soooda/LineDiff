import torch
import torch.nn as nn
import torchvision

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class VGG19(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(VGG19, self).__init__()
        if pretrained:
            model = torchvision.models.vgg19(weights='IMAGENET1K_V1')
        else:
            model = torchvision.models.vgg19()
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(4096, num_classes) # Change output layer size
        self.model = model
        self.features = model.features

    def forward(self, x):
        return self.model(x)


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGPerceptualLoss(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = VGG19(num_classes=num_classes, pretrained=pretrained)
        self.vgg_pretrained_features = self.vgg.features
        self.vgg_pretrained_features.to(device)
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        # Freeze the weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=[2, 7, 12, 21, 30]):
        X = self.normalize(X)
        Y = self.normalize(Y)
        weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5] # Alphas
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss

    def load_weight(self, weight_path):
        """Load weights into the internal VGG19 model."""
        state_dict = torch.load(weight_path, map_location=device)
        self.vgg.load_state_dict(state_dict['state_dict'], strict=False)
        self.vgg.eval()  # Make sure model is in eval mode
        for param in self.vgg.parameters():
            param.requires_grad = False
