import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaborConv2d(nn.Module):
    """
    A convolutional layer initialized with Gabor filters.
    The filters can be fixed or learnable.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(GaborConv2d, self).__init__()
        self.is_calculated = False

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.kernel_size = kernel_size

        # Initialize weights with Gabor filters
        self.initialize_gabor_filters(out_channels, in_channels, kernel_size, groups)

    def initialize_gabor_filters(self, out_channels, in_channels, kernel_size, groups):
        # Generate Gabor filters
        filters = []
        for i in range(out_channels):
            # Vary parameters for each filter to capture different orientations/frequencies
            theta = (i / out_channels) * math.pi 
            sigma = 1.0
            lambd = 10.0
            gamma = 0.5
            psi = 0

            kernel = self.get_gabor_kernel(kernel_size, sigma, theta, lambd, gamma, psi)
            # Normalize
            kernel = kernel / kernel.sum()
            filters.append(kernel)
        
        filters = torch.stack(filters) # [out_channels, kernel_size, kernel_size]
        filters = filters.unsqueeze(1) # [out_channels, 1, kernel_size, kernel_size]
        
        # Adjust for groups
        # Conv2d weight shape: [out_channels, in_channels / groups, k, k]
        channels_per_group = in_channels // groups
        if channels_per_group > 1:
            filters = filters.repeat(1, channels_per_group, 1, 1)
        
        # Assign to conv layer
        self.conv_layer.weight.data = filters
        
        # Optional: Freeze weights to keep them as fixed Gabor filters
        # self.conv_layer.weight.requires_grad = False

    def get_gabor_kernel(self, ksize, sigma, theta, lambd, gamma, psi):
        sigma_x = sigma
        sigma_y = sigma / gamma
        xmax = ksize // 2
        ymax = ksize // 2
        xmin = -xmax
        ymin = -ymax
        (y, x) = torch.meshgrid(torch.arange(ymin, ymax + 1), torch.arange(xmin, xmax + 1))
        
        # Rotation
        x_theta = x * math.cos(theta) + y * math.sin(theta)
        y_theta = -x * math.sin(theta) + y * math.cos(theta)

        gb = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos(2 * math.pi / lambd * x_theta + psi)
        return gb

    def forward(self, x):
        return self.conv_layer(x)
