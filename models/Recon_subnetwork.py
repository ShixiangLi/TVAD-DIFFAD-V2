# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb): # emb will be the combined_embed
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            # downsamples by 1/2
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, combined_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, combined_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """
    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert (
                    in_channels % n_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {n_head_channels}"
            self.num_heads = in_channels // n_head_channels

        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, combined_embed=None):
        b, c, *spatial = x.shape
        x_norm = self.norm(x)
        x_reshaped = x_norm.reshape(b, c, -1)
        qkv = self.to_qkv(x_reshaped)
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x.reshape(b, c, -1) + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, combined_embed=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ResBlock(TimestepBlock):
    def __init__(
            self,
            in_channels,
            embed_dim,
            dropout,
            out_channels=None,
            use_conv=False,
            up=False,
            down=False
            ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_layers = nn.Sequential(
                GroupNorm32(32, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, 3, padding=1)
                )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, out_channels)
                )
        self.out_layers = nn.Sequential(
                GroupNorm32(32, out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
                )
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, combined_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x_skip = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            x_skip = x

        emb_out = self.embed_layers(combined_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x_skip) + h


class UNetModel(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            conv_resample=True,
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="32,16,8",
            biggan_updown=True,
            in_channels=1,
            ):
        super().__init__()

        if isinstance(img_size, int):
            img_size_tuple = (img_size, img_size)
        else:
            img_size_tuple = img_size

        if channel_mults == "":
            if img_size_tuple[0] == 64:
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size_tuple[0]}")
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size_tuple[0] // int(res))

        self.image_size = img_size_tuple
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(base_channels, 1),
                nn.Linear(base_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                )
        
        # MODIFIED: Replaced nn.Linear with a 1D CNN to process the time-series data.
        self.current_feature_embedding_layer = nn.Sequential(
            # Input shape will be (N, 3, 24) after permutation
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), # Global Average Pooling collapses the time dimension
            nn.Flatten(), # Flattens the output to (N, 128)
            nn.Linear(128, time_embed_dim) # Project to the same dimension as time embedding
        )

        # Input block
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv2d(self.in_channels, self.model_channels, 3, padding=1))]
                )
        
        ch = self.model_channels
        channels_list = [ch]
        ds = 1

        # Downsampling path
        for i, mult in enumerate(channel_mults):
            out_ch = self.model_channels * int(mult)
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch,
                        embed_dim=time_embed_dim,
                        out_channels=out_ch,
                        dropout=dropout,
                        )]
                ch = out_ch
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels_list.append(ch)

            if i != len(channel_mults) - 1:
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(
                                        ch,
                                        embed_dim=time_embed_dim,
                                        out_channels=ch,
                                        dropout=dropout,
                                        down=True
                                        )
                                if biggan_updown
                                else
                                Downsample(ch, self.conv_resample, out_channels=ch)
                                )
                        )
                ds *= 2
                channels_list.append(ch)

        # Middle block
        self.middle = TimestepEmbedSequential(
                ResBlock(ch, embed_dim=time_embed_dim, dropout=dropout),
                AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels),
                ResBlock(ch, embed_dim=time_embed_dim, dropout=dropout)
                )
        
        # Upsampling path
        self.up = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = self.model_channels * int(mult)
            for j in range(num_res_blocks + 1):
                skip_ch = channels_list.pop()
                layers = [
                    ResBlock(
                            ch + skip_ch,
                            embed_dim=time_embed_dim,
                            out_channels=out_ch,
                            dropout=dropout
                            )
                    ]
                ch = out_ch

                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels),
                            )

                if i != 0 and j == num_res_blocks:
                    layers.append(
                            ResBlock(
                                    ch,
                                    embed_dim=time_embed_dim,
                                    out_channels=ch,
                                    dropout=dropout,
                                    up=True
                                    )
                            if biggan_updown
                            else
                            Upsample(ch, self.conv_resample, out_channels=ch)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        # Output block
        self.out = nn.Sequential(
                GroupNorm32(32, ch),
                nn.SiLU(),
                zero_module(nn.Conv2d(ch, self.out_channels, 3, padding=1))
                )

    def forward(self, x, time, current_features): # current_features shape: (N, 24, 3)
        time_embed = self.time_embedding(time)
        
        # MODIFIED: Permute and pass current_features through the new CNN layer
        # (N, 24, 3) -> (N, 3, 24) for Conv1d
        current_features_transposed = current_features.permute(0, 2, 1)
        current_embed = self.current_feature_embedding_layer(current_features_transposed)

        # Now both embeddings have a compatible shape for addition
        combined_embed = time_embed + current_embed 

        skips = []
        h = x.type(self.dtype)
        for module in self.down:
            h = module(h, combined_embed)
            skips.append(h)
        
        h = self.middle(h, combined_embed)
        
        for module in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, combined_embed)
            
        h = h.type(x.dtype)
        return self.out(h)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)


if __name__ == "__main__":
    args = {
        'img_size':          (64, 64),
        'base_channels':     32,
        'dropout':           0,
        'num_heads':         4,
        'attention_resolutions': "8,4",
        'num_head_channels': -1,
        'channel_mults': (1, 2, 3, 4),
        'in_channels':       3,
        'Batch_Size':        64
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetModel(
            img_size=args['img_size'], 
            base_channels=args['base_channels'], 
            channel_mults=args['channel_mults'],
            dropout=args["dropout"], 
            n_heads=args["num_heads"], 
            n_head_channels=args["num_head_channels"],
            attention_resolutions=args["attention_resolutions"],
            in_channels=args['in_channels']
            ).to(device)

    batch_size = args['Batch_Size']
    dummy_x = torch.randn(batch_size, args['in_channels'], args['img_size'][0], args['img_size'][1]).to(device)
    dummy_t = torch.randint(0, 1000, (batch_size,), device=device).float()
    # MODIFIED: Update dummy tensor shape for current_features for testing
    dummy_current_features = torch.randn(batch_size, 24, 3).to(device)
    
    print("Input x shape:", dummy_x.shape)
    print("Input time shape:", dummy_t.shape)
    print("Input current_features shape:", dummy_current_features.shape)
    
    output = model(dummy_x, dummy_t, dummy_current_features)
    print("Output shape:", output.shape)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")