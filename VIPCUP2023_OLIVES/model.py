import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class ResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=2):
        super(ResNet, self).__init__()
        if (name == 'resnet50'):
            self.encoder = torchvision.models.resnet50(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.fc(self.encoder(x))

class VGGNet(nn.Module):
    """VGG encoder + classifier"""
    def __init__(self, name='vgg16', num_classes=2):
        super(VGGNet, self).__init__()
        if name == 'vgg16':
            self.encoder = models.vgg16(pretrained=True)
            self.encoder.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Adjust input channels
            self.encoder.classifier = nn.Identity()  # Remove original classifier
            self.fc = nn.Linear(512, num_classes)  # Custom classifier

        else:
            raise ValueError("Only 'vgg16' and 'vgg19' are supported.")

    def forward(self, x):
        x = self.encoder.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class InceptionNet(nn.Module):
    """Inception encoder + classifier"""
    def __init__(self, num_classes=2):
        super(InceptionNet, self).__init__()
        self.encoder = models.inception_v3(pretrained=True, aux_logits=False)
        self.encoder.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)  # Adjust input channels
        self.encoder.fc = nn.Identity()  # Remove original classifier
        self.fc = nn.Linear(2048, num_classes)  # Custom classifier

    def forward(self, x):
        # Resize input to 299x299 as required by Inception
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = self.encoder(x)
        return self.fc(x)
