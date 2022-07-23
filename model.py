import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 mode='conv', use_relu=True, ReLU_slope=0.2, use_dropout=False, 
                 dropout_p=0.5, use_batchnorm = True):
        super().__init__()
        
        conv_args = (
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode
        )

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_relu = use_relu
        self.mode = mode

        if self.use_batchnorm:
            self.bias = False
        else:
            self.bias = True

        available_modes = ['conv', 'deconv']
        if not self.mode in available_modes:
            print(f"{self.mode} is not correct; correct modes: {available_modes}")
            raise NotImplementedError

        if self.mode == 'conv':
            self.conv = nn.Conv2d(*conv_args)
        else:
            self.conv = nn.ConvTranspose2d(*conv_args)

        if use_dropout:
            self.dropout = nn.Dropout(dropout_p) 
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.LeakyReLU(ReLU_slope)


    def forward(self, *args, **kwargs):
        out = self.conv(*args, **kwargs)

        if self.use_batchnorm:
            out = self.bn(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.use_relu:
            out = self.relu(out)
            
        return out
        
        
class PositionalEmbedding(nn.Module):
    """
    Positional encodings.
    For more information: https://arxiv.org/abs/1706.037624
    
    Note that embeddings are arranged like [sin, sin, ..., sin, cos, ..., cos], not [sin, cos, ..., sin, cos].
    
    Parameters
    ----------
    dim : Number of embedding dimensions.
    
    Input
    ----------
    pos : 1-dimensional torch.Tensor of positions.
    
    Output
    ----------
    embedding : 2-dimensional torch.Tensor with shape (len(pos), dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    
    def forward(self, pos):
        d = self.dim if self.dim%2 == 0 else self.dim+1
        
        idxs = torch.arange(0.0, d/2, device=pos.device)/self.dim
        
        sin_arg = pos.view(-1, 1)/10000**(idxs.view(1, -1))
        cos_arg = sin_arg if self.dim%2 == 0 else sin_arg[:, :-1]
        embedding = torch.cat([torch.sin(sin_arg), torch.cos(cos_arg)], dim=1)
        
        return embedding
    
    
class UpsampleLayer(nn.Module):
    def __init__(self, upsample, conv):
        super().__init__()
        self.upsample = upsample
        self.conv = conv
        
    def forward(self, input, residual_input):
        output = self.upsample(input)
        output = self.conv(torch.cat([output, residual_input], dim=1))
        
        return output
        
        
class DiffusionNet(nn.Module):
    def __init__(self, channels=3, last_activation=nn.Tanh()):
        super().__init__()
        self.last_activation = last_activation
        self.positional_embedding_dim = 32
        self.pos_emb = PositionalEmbedding(self.positional_embedding_dim)
        
        channel_dims = [channels, 32, 64, 128, 256, 512]

        self.conv0 = ConvLayer(channel_dims[0], channel_dims[1], 3, stride=2, padding=1)  # 64 -> 32
        self.conv1 = ConvLayer(channel_dims[1]+self.positional_embedding_dim,
                                                channel_dims[2], 3, stride=2, padding=1)  # 32 -> 16
        self.conv2 = ConvLayer(channel_dims[2], channel_dims[3], 3, stride=2, padding=1)  # 16 ->  8
        self.conv3 = ConvLayer(channel_dims[3], channel_dims[4], 3, stride=2, padding=1)  #  8 ->  4
        self.conv4 = ConvLayer(channel_dims[4], channel_dims[5], 3, stride=2, padding=1)  #  4 ->  2
        
        self.upconv4 = UpsampleLayer(
            nn.Upsample(scale_factor=2, mode='nearest'),  #  2 ->  4
            ConvLayer(channel_dims[5]+channel_dims[4], channel_dims[4], 3, padding=1)
        )
        self.upconv3 = UpsampleLayer(
            nn.Upsample(scale_factor=2, mode='nearest'),  #  4 ->  8
            ConvLayer(channel_dims[4]+channel_dims[3], channel_dims[3], 3, padding=1)
        )
        self.upconv2 = UpsampleLayer(
            nn.Upsample(scale_factor=2, mode='nearest'),  #  8 -> 16
            ConvLayer(channel_dims[3]+channel_dims[2], channel_dims[2], 3, padding=1)
        )
        self.upconv1 = UpsampleLayer(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16 -> 32
            ConvLayer(channel_dims[2]+channel_dims[1]+self.positional_embedding_dim, channel_dims[1], 3, padding=1)
        )
        self.upconv0 = UpsampleLayer(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32 -> 64
            ConvLayer(channel_dims[1]+channel_dims[0], channel_dims[0], 3, padding=1,
                      use_relu=False, use_batchnorm=False)
        )

        
    def forward(self, input, t):
        conv0 = self.conv0(input)
        pos_emb = self.pos_emb(t)[:, :, None, None].expand(-1, -1, conv0.shape[2], conv0.shape[3])
        
        conv0 = torch.cat([conv0, pos_emb], dim=1)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        output = self.conv4(conv3)
        
        output = self.upconv4(output, conv3)
        output = self.upconv3(output, conv2)
        output = self.upconv2(output, conv1)
        output = self.upconv1(output, conv0)
        output = self.upconv0(output, input)

        output = self.last_activation(output)
        
        return output
    