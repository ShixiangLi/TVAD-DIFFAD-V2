# models/aligner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class StandardResBlock(nn.Module):
    """
    A standard, non-conditional ResNet block.
    This is self-contained and does not depend on the diffusion model's ResBlock.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(StandardResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to handle changes in dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ImageEncoder(nn.Module):
    """
    Convolutional Image Encoder using standard ResBlocks.
    Input: (N, 3, 64, 64)
    Output: (N, latent_dim)
    """
    def __init__(self, base_channels=64, channel_mults=(1, 2, 4, 8), latent_dim=256):
        super().__init__()
        
        layers = [nn.Conv2d(8, base_channels, 3, padding=1)]
        ch = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            stride = 2 if i > 0 else 1
            layers.append(StandardResBlock(ch, out_ch, stride=stride))
            layers.append(StandardResBlock(out_ch, out_ch))
            ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = nn.Linear(ch, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        h = self.final_pool(h)
        h = h.view(h.shape[0], -1)
        return self.projection_head(h)

class CurrentEncoder(nn.Module):
    """
    Transformer-based Current Data Encoder.
    Input: (N, 24, 3)
    Output: (N, latent_dim)
    """
    def __init__(self, input_dim=3, model_dim=128, n_heads=4, num_encoder_layers=3, latent_dim=256):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, batch_first=True, dim_feedforward=model_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        
        self.projection_head = nn.Linear(model_dim, latent_dim)

    def forward(self, x):
        x = self.input_projection(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        encoded = self.transformer_encoder(x)
        cls_output = encoded[:, 0, :]
        return self.projection_head(cls_output)

class Aligner(nn.Module):
    """
    The main Aligner model.
    """
    def __init__(self, image_encoder, current_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.current_encoder = current_encoder

    def forward(self, image, current):
        image_embedding = self.image_encoder(image)
        current_embedding = self.current_encoder(current)
        
        # L2 normalize embeddings for cosine similarity.
        # Added eps=1e-8 for numerical stability to prevent division by zero.
        image_embedding = F.normalize(image_embedding, p=2, dim=1, eps=1e-8)
        current_embedding = F.normalize(current_embedding, p=2, dim=1, eps=1e-8)
        
        return image_embedding, current_embedding