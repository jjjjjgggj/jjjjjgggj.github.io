import math
import torch
from torch import nn
from inspect import isfunction
from einops.layers.torch import Rearrange, Reduce
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta= self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            #x = (1 + gamma) * x + beta
            x = x + 0.5 * (gamma + beta) * x + 0.5 * (gamma + beta)
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Sigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=8, mode="nearest")
        self.up = nn.Upsample(scale_factor=4, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Upsample1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Upsample2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=1, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module): #原
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=True, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        if exists(time_emb):
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        if exists(time_emb):
            x = self.res_block(x, time_emb)
        else:
            x = self.res_block(x, None)
        x = self.attn(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, n_div=12, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        # self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.q = nn.Conv2d(in_channel, in_channel//n_div, 1, bias=False)
        self.kv = nn.Conv2d(in_channel, in_channel//n_div * 2, 1, bias=False)
        # self.out = nn.Conv2d(in_channel, in_channel//n_div, 1)
        self.out = nn.Conv2d(in_channel//n_div, in_channel, 3, padding=1)

    def forward(self, x, xe, n_div=12):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // (n_head * n_div)
        # x = self.norm(x)
        # xe = self.norm(xe)
        query = self.q(x).view(batch, n_head, head_dim * 1, height, width)
        kv = self.kv(xe).view(batch, n_head, head_dim * 2, xe.shape[-2], xe.shape[-1])
        key, value = kv.chunk(2, dim=2)
        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel//n_div)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel//n_div, height, width))

        return out + x

class ResnetBlocWithCrossAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, cross_attn_n_head=1, cross_attn_n_div=12):
        super().__init__()
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.attn = CrossAttention(dim_out, n_head=cross_attn_n_head, n_div=cross_attn_n_div)

    def forward(self, x, xe, time_emb):
        if exists(time_emb):
            x = self.res_block(x, time_emb)
        else:
            x = self.res_block(x, None)
        x = self.attn(x, xe)
        return x


class ResnetBlocks(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)

    def forward(self, x, time_emb):
        if exists(time_emb):
            x = self.res_block(x, time_emb)
        else:
            x = self.res_block(x, None)
        return x


class Net(nn.Module):
    def __init__(
            self,
            in_channel=1,
            out_channel=1,
            inner_channel=24,
            norm_groups=32,
            channel_mults=(1, 4, 16),
            attn_res=2,
            res_blocks=2,
            dropout=0.1,
            with_noise_level_emb=True,
            image_size=112
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        fusions = []
        pre_channel = inner_channel  # 24
        feat_channels = [24]  # [24]
        downs = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]


        fusions_ultra = []
        pre_channel_ultra = inner_channel
        feat_channels_ultra = [24]
        downs_ultra = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]


        channel_mult = inner_channel * channel_mults[0]  # 24 * 1 = 24
        # downs: 第一层, pre_channel = 24 , channel_mult = 24
        downs.append(ResnetBlocWithAttn(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24]
        pre_channel = channel_mult  # 24
        # pre_channel = 24 , channel_mult = 24
        downs.append(ResnetBlocks(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24,24]
        pre_channel = channel_mult  # 24
        fusions.append(Upsample(pre_channel))

        downs.append(Downsample(pre_channel))
        feat_channels.append(pre_channel)  # [24,24,24,24]
        channel_mult = inner_channel * channel_mults[1]  # 24 * 4 = 96
        # downs: 第二层, pre_channel = 24 , channel_mult = 96
        downs.append(ResnetBlocWithAttn(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24,24,24,96]
        pre_channel = channel_mult  # 96
        # pre_channel = 96 , channel_mult = 96
        downs.append(ResnetBlocks(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24,24,24,96,96]
        pre_channel = channel_mult  # 96
        fusions.append(Upsample(pre_channel))

        downs.append(Downsample(pre_channel))
        feat_channels.append(pre_channel)  # [24,24,24,24,96,96,96]
        channel_mult = inner_channel * channel_mults[2]  # 24 * 16 = 384
        # downs: 第三层, pre_channel = 96 ,channel_mult = 384
        downs.append(ResnetBlocWithAttn(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24,24,24,96,96,96,384]
        pre_channel = channel_mult  # 384
        # pre_channel = 384 , channel_mult = 384
        downs.append(ResnetBlocks(
            pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))
        feat_channels.append(channel_mult)  # [24,24,24,24,96,96,96,384,384]
        #fusions.append(Upsample(pre_channel))

        self.downs = nn.ModuleList(downs)
        self.fusions = nn.ModuleList(fusions)

        mids = []
        for _ in range(0, attn_res):
            mids.append(ResnetBlocks(  # pre_channel = 384
                pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                norm_groups=min(norm_groups, pre_channel // 8),
                dropout=dropout))
            mids.append(CrossAttention(pre_channel, norm_groups=min(norm_groups, pre_channel // 8)))
        self.mid = nn.ModuleList(mids)

        ups = []
        channel_mult = inner_channel * channel_mults[2]  # 24 * 16 = 384
        # ups:第三层 pre_channel = 384, feat_channels.pop() = 384, channel_mult = 384
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),  # norm_groups=32, pre_channel // 8 = 48
            dropout=dropout))  # [24,24,24,24,96,96,96,384]
        pre_channel = channel_mult  # 384
        ups.append(ResnetBlocWithAttn(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24,24,24,96,96,96]
        pre_channel = channel_mult  # 384
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),  # norm_groups=32, pre_channel // 8 = 48
            dropout=dropout))
        pre_channel = channel_mult
        ups.append(Upsample1(pre_channel))

        channel_mult = inner_channel * channel_mults[1]  # 24 * 4 =96
        # ups:第二层 pre_channel = 384, feat_channels.pop() = 96, channel_mult = 96
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop() , channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24,24,24,96,96]
        pre_channel = channel_mult  # 96
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24,24,24,96]
        pre_channel = channel_mult  # 96
        ups.append(ResnetBlocWithAttn(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24,24,24]
        pre_channel = channel_mult  # 96
        ups.append(Upsample1(pre_channel))

        channel_mult = inner_channel * channel_mults[0]  # 24 * 1 =24
        # ups:第一层 pre_channel = 96, feat_channels.pop() = 24, channel_mult = 24
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop() , channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24,24]
        pre_channel = channel_mult  # 24
        ups.append(ResnetBlocks(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24,24]
        pre_channel = channel_mult  # 96
        ups.append(ResnetBlocWithAttn(
            pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel // 8),
            dropout=dropout))  # [24]
        pre_channel = channel_mult  # 96

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel),
                                groups=min(norm_groups, pre_channel // 8))


        channel_mult_ultra = inner_channel * channel_mults[0]  # 24 * 1 = 24
        # downs: 第一层, pre_channel_ultra = 24 , channel_mult_ultra = 24
        downs_ultra.append(ResnetBlocWithAttn(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24]
        pre_channel_ultra = channel_mult_ultra  # 24
        # pre_channel_ultra = 24 , channel_mult_ultra = 24
        downs_ultra.append(ResnetBlocks(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24,24]
        pre_channel_ultra = channel_mult_ultra  # 24
        fusions_ultra.append(Upsample(pre_channel_ultra))

        downs_ultra.append(Downsample(pre_channel_ultra))
        feat_channels_ultra.append(pre_channel_ultra)  # [24,24,24,24]
        channel_mult_ultra = inner_channel * channel_mults[1]  # 24 * 4 = 96
        # downs: 第二层, pre_channel_ultra = 24 , channel_mult_ultra = 96
        downs_ultra.append(ResnetBlocWithAttn(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24,24,24,96]
        pre_channel_ultra = channel_mult_ultra  # 96
        # pre_channel_ultra = 96 , channel_mult_ultra = 96
        downs_ultra.append(ResnetBlocks(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24,24,24,96,96]
        pre_channel_ultra = channel_mult_ultra  # 96
        fusions_ultra.append(Upsample(pre_channel_ultra))

        downs_ultra.append(Downsample(pre_channel_ultra))
        feat_channels_ultra.append(pre_channel_ultra)  # [24,24,24,24,96,96,96]
        channel_mult_ultra = inner_channel * channel_mults[2]  # 24 * 16 = 384
        # downs: 第三层， pre_channel_ultra = 96, channel_mult_ultra = 384
        downs_ultra.append(ResnetBlocWithAttn(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24,24,24,96,96,96,384]
        pre_channel_ultra = channel_mult_ultra  # 384
        # pre_channel_ultra = 384 , channel_mult_ultra = 384
        downs_ultra.append(ResnetBlocks(
            pre_channel_ultra, channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))
        feat_channels_ultra.append(channel_mult_ultra)  # [24,24,24,24,96,96,96,384,384]
        pre_channel_ultra = channel_mult_ultra  # 384
        #fusions_ultra.append(Upsample(pre_channel_ultra))

        self.downs_ultra = nn.ModuleList(downs_ultra)
        self.fusions_ultra = nn.ModuleList(fusions_ultra)

        mids_ultra = []
        for _ in range(0, attn_res):
            mids_ultra.append(ResnetBlocks(  # pre_channel = 384
                pre_channel_ultra, pre_channel_ultra, noise_level_emb_dim=noise_level_channel,
                norm_groups=min(norm_groups, pre_channel_ultra // 8),
                dropout=dropout))
            mids_ultra.append(CrossAttention(pre_channel_ultra, norm_groups=min(norm_groups, pre_channel_ultra // 8)))
        self.mid_ultra = nn.ModuleList(mids_ultra)

        ups_ultra = []
        channel_mult_ultra = inner_channel * channel_mults[2]  # 24 * 16 = 384
        pre_channel_ultra = 384
        #ups:第三层 pre_channel_ultra = 384, feat_channels_ultra.pop() = 384, channel_mult_ultra = 384
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 384 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),  #norm_groups=32, pre_channel // 8 = 48
            dropout=dropout)) #[24,24,24,24,96,96,96,384]
        pre_channel_ultra = channel_mult_ultra  # 384
        ups_ultra.append(ResnetBlocWithCrossAttn(
            pre_channel_ultra + feat_channels_ultra.pop() + 384  , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout)) #[24,24,24,24,96,96,96]
        pre_channel_ultra = channel_mult_ultra # 384
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 384 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),  # norm_groups=32, pre_channel // 8 = 48
            dropout=dropout)) #[24,24,24,24,96,96]
        pre_channel_ultra = channel_mult_ultra
        ups_ultra.append(Upsample1(pre_channel_ultra))

        channel_mult_ultra = inner_channel * channel_mults[1]  # 24 * 4 =96
        # ups:第二层 pre_channel_ultra = 96, feat_channels_ultra.pop() = 96, channel_mult_ultra = 96
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 96 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [24,24,24,24,96]
        pre_channel_ultra = channel_mult_ultra  # 96
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 96 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [24,24,24,24]
        pre_channel_ultra = channel_mult_ultra  # 96
        ups_ultra.append(ResnetBlocWithCrossAttn(
            pre_channel_ultra + feat_channels_ultra.pop() + 96 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [24,24,24]
        pre_channel_ultra = channel_mult_ultra  # 96
        ups_ultra.append(Upsample1(pre_channel_ultra))

        channel_mult_ultra = inner_channel * channel_mults[0]  # 24 * 1 =24
        # ups:第一层 pre_channel_ultra = 96, feat_channels_ultra.pop() = 24, channel_mult_ultra = 24
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 24 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [24,24]
        pre_channel_ultra = channel_mult_ultra  # 24
        ups_ultra.append(ResnetBlocks(
            pre_channel_ultra + feat_channels_ultra.pop() + 24 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [24]
        pre_channel_ultra = channel_mult_ultra  # 24
        ups_ultra.append(ResnetBlocWithCrossAttn(
            pre_channel_ultra + feat_channels_ultra.pop() + 24 , channel_mult_ultra, noise_level_emb_dim=noise_level_channel, norm_groups=min(norm_groups, pre_channel_ultra // 8),
            dropout=dropout))  # [ ]
        pre_channel_ultra = channel_mult_ultra  # 24

        self.ups_ultra = nn.ModuleList(ups_ultra)

        self.final_conv_ultra = Block(pre_channel_ultra, default(out_channel, in_channel),
                                groups=min(norm_groups, pre_channel_ultra // 8))

    def forward(self, x, xe, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        n = 0
        n_ultra = 0

        feats = []
        feats_ultra = []
        feats_ups = []
        reversed_feats_ups = []

        x_ultra = x

        nums_rb = 0

        for layer in self.downs:
            if isinstance(layer, ResnetBlocks):
                x = layer(x, t)
                xe = layer(xe, None)
                if n == 0 or n == 1:
                    x = x + self.fusions[n](xe)
                n = n + 1
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                xe = layer(xe, None)
            else:
                x = layer(x)
                xe = layer(xe)

            feats.append(x)


        for layer in self.mid:
            if isinstance(layer, ResnetBlocks):
                x = layer(x, t)
                xe = layer(xe, None)
            else:
                x = layer(x, xe)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocks):
                if nums_rb == 2:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                elif nums_rb == 4:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                else:
                    x = layer(torch.cat((x, feats.pop()), dim=1), t)
                nums_rb = nums_rb + 1
                feats_ups.append(x)
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                feats_ups.append(x)
            else:
                x = layer(x)

        x = self.final_conv(x)

        while feats_ups:
            reversed_feats_ups.append(feats_ups.pop())

        for layer in self.downs_ultra:
            if isinstance(layer, ResnetBlocks):
                x_ultra = layer(x_ultra, t)
                x = layer(x, None)
                if n_ultra == 0 or n_ultra == 1 :
                    x_ultra = x_ultra + x
                n_ultra = n_ultra + 1
            elif isinstance(layer, ResnetBlocWithAttn):
                x_ultra = layer(x_ultra, t)
                x = layer(x, None)
            else:
                x_ultra = layer(x_ultra)
                x = layer(x)

            feats_ultra.append(x_ultra)


        for layer in self.mid_ultra:
            if isinstance(layer, ResnetBlocks):
                x_ultra = layer(x_ultra, t)
                x = layer(x, None)
            else:
                x_ultra = layer(x_ultra, x)

        for layer in self.ups_ultra:
            if isinstance(layer, ResnetBlocks):
                x_ultra = layer(torch.cat((x_ultra, feats_ultra.pop(), reversed_feats_ups.pop()), dim=1), t)
            elif isinstance(layer, ResnetBlocWithCrossAttn):
                con = reversed_feats_ups.pop()
                x_ultra = layer(torch.cat((x_ultra, feats_ultra.pop(), con ), dim=1), con, t)
            else:
                x_ultra = layer(x_ultra)

        return self.final_conv_ultra(x_ultra)


# if __name__ == '__main__':
#     net = Net24WP(
#         in_channel=1,
#         out_channel=1,
#         norm_groups=32,
#         inner_channel=24,
#         channel_mults=[1,4,8,16],
#         attn_res=2,
#         res_blocks=2,
#         dropout=0,
#         image_size=112,
#         with_noise_level_emb=True
#     )
#     x = torch.randn(1,1,112,112)
#     xe = torch.randn(1,1,28,28)
#     #t = torch.tensor([1013])
#     t = torch.tensor([[0.5740]])
#     out = net(x, xe, t)
#     print(net)