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


from torchvision.models import vit_b_16, ViT_B_16_Weights  # Vision Transformer (ViT) base model
import torch.nn.init as init
class Transformer(nn.Module):
    """Transformer-based encoder + classifier"""
    def __init__(self, name='vit_b_16', num_classes=2):
        super(Transformer, self).__init__()
        self.name = name
        if name == 'vit_b_16':  # Vision Transformer Base (16x16 patches)
            # Load Vision Transformer with pretrained weights
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            # Adjust the input layer for grayscale (1-channel) input
            self.encoder.conv_proj = nn.Conv2d(
                1, 768, kernel_size=(16, 16), stride=(16, 16))
            # Remove the classification head
            self.encoder.heads = nn.Identity()
            # Add a custom classification head
            self.fc = nn.Linear(768, num_classes)

        elif name in ['maxvit_base_tf_224', 'maxvit_tiny_tf_224']:  # MaxViT
            print(f"Using model {name}")
            name = 'maxvit_tiny_tf_224'
            # Load the pretrained MaxViT model
            self.encoder = timm.create_model(name, pretrained=True)
            if self.encoder.stem.conv1.bias is not None:
                init.constant_(self.encoder.stem.conv1.bias, 0)
            # Remove the classification head
            self.encoder.head = nn.Identity()
            # Define a new classifier head
            self.fc = nn.Linear(self.encoder.num_features, num_classes)
        elif name == 'eva02_base_patch16_clip_224':  # Small EVA model with 16x16 patches, input size 224x224
            print("Using EVA-02 Small Patch 224")
            self.encoder = timm.create_model(name, pretrained=True)
            self.encoder.patch_embed.proj = nn.Conv2d(1, self.encoder.patch_embed.proj.out_channels, kernel_size=(16, 16), stride=(16, 16))
            self.encoder.head = nn.Identity()  # Remove existing classification head
            self.fc = nn.Linear(self.encoder.num_features, num_classes)  # New classification head
        else:
            raise ValueError(f"Unknown model name: {name}")
    
    def forward(self, x):
        features = self.encoder(x)  # Extract features
        if self.name in ['maxvit_base_tf_224', 'maxvit_tiny_tf_224']:
            # need a final pooling layer for maxvit
            features = features.mean(dim=[2, 3])+1e-8
        return self.fc(features)   # Classification head
       