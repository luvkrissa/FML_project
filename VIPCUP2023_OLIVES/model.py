import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision

class ResNet(nn.Module):
    def __init__(self, name='resnet50', num_classes=2, clinical_input_dim=4, fusion_dim=256):
        super(ResNet, self).__init__()
        
        if name == 'resnet50':
            self.encoder = torchvision.models.resnet50(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            image_feature_dim = 2048
        elif name == 'resnet101':
            self.encoder = torchvision.models.resnet101(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            image_feature_dim = 2048
        elif name == 'resnet152':
            self.encoder = torchvision.models.resnet152(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            image_feature_dim = 2048
        elif name == 'inceptionresnetv2':
            self.encoder = timm.create_model("inception_resnet_v2", pretrained=False)
            self.encoder.conv2d_1a.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False) 
            self.encoder.classif = nn.Identity()
            image_feature_dim = 1536
        else:  # Default to ResNet18
            self.encoder = torchvision.models.resnet18(zero_init_residual=True)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.fc = nn.Identity()
            image_feature_dim = 512

        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, fusion_dim) 
        )

        self.fusion_layer = nn.Linear(image_feature_dim + fusion_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, x_image, x_clinical):
        image_features = self.encoder(x_image)

        clinical_features = self.clinical_branch(x_clinical)

        fused_features = torch.cat((image_features, clinical_features), dim=1)
        fused_features = self.fusion_layer(fused_features)

        output = self.classifier(fused_features)
        return output
