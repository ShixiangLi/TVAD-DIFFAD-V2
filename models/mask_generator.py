# models/mask_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicMaskGenerator(nn.Module):
    def __init__(self, threshold=0.1, blur_radius=5.0):
        super().__init__()
        self.threshold = threshold
        self.blur_radius = blur_radius
        
    def custom_gaussian_blur(self, x, kernel_size, sigma):
        """
        自定义高斯模糊实现，不依赖torchvision
        
        Args:
            x: 输入张量 [B, C, H, W]
            kernel_size: 高斯核大小
            sigma: 高斯核标准差
        """
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        
        # 计算一维高斯核
        channels = x.shape[1]
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        
        # 计算高斯核
        meshgrids = torch.meshgrid([torch.arange(size, device=x.device, dtype=torch.float32) for size in kernel_size], indexing='ij')
        
        # 中心点坐标
        center = [(size - 1) / 2 for size in kernel_size]
        
        # 计算二维高斯核
        gauss = torch.ones(1, device=x.device)
        for grid, sigma_value, center_value in zip(meshgrids, sigma, center):
            gauss = gauss * torch.exp(-((grid - center_value) / sigma_value) ** 2 / 2)
        
        # 归一化高斯核
        gauss = gauss / gauss.sum()
        
        # 扩展到所需维度 [1, 1, kernel_size, kernel_size]
        gauss = gauss.view(1, 1, kernel_size[0], kernel_size[1])
        gauss = gauss.repeat(channels, 1, 1, 1)
        
        # 创建卷积层
        pad_size = kernel_size[0] // 2
        padding = [pad_size, pad_size]
        
        # 对每个通道单独做卷积
        result = []
        for c in range(channels):
            channel = x[:, c:c+1, :, :]  # [B, 1, H, W]
            blurred = F.conv2d(channel, gauss[c:c+1], padding=padding, groups=1)
            result.append(blurred)
            
        return torch.cat(result, dim=1)
        
    def forward(self, original, reconstructed):
        """
        基于原始图像和初步重建图像的差异生成掩码
        
        Args:
            original: 原始图像 (B, C, H, W)
            reconstructed: 初步重建图像 (B, C, H, W)
        
        Returns:
            mask: 指示潜在异常区域的掩码 (B, 1, H, W)，1表示异常区域
        """
        # 计算重建误差
        diff = torch.abs(original - reconstructed)
        
        # 计算每个像素点的误差总和
        if diff.shape[1] > 1:  # 如果是多通道图像
            diff_sum = diff.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            diff_sum = diff
        
        # 应用高斯模糊使差异更平滑
        kernel_size = int(self.blur_radius * 2) + 1
        diff_blur = self.custom_gaussian_blur(diff_sum, kernel_size, self.blur_radius)
        
        # 对每个样本独立进行归一化
        batch_size = diff_blur.shape[0]
        normalized_diff = torch.zeros_like(diff_blur)
        
        for i in range(batch_size):
            sample_diff = diff_blur[i]
            min_val = sample_diff.min()
            max_val = sample_diff.max()
            if max_val > min_val:
                normalized_diff[i] = (sample_diff - min_val) / (max_val - min_val)
            else:
                normalized_diff[i] = torch.zeros_like(sample_diff)
        
        # 生成二值掩码
        mask = (normalized_diff > self.threshold).float()
        
        return mask 