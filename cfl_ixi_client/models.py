"""The Fed-IXI 3D U-Net, built from the pytorch-3dunet building blocks.

Lineage
-------
* Architecture family: Cicek, Abdulkadir, Lienkamp, Brox & Ronneberger, "3D U-Net:
  Learning Dense Volumetric Segmentation from Sparse Annotation", arXiv:1606.06650.
  Its conventions are kept: per level two 3x3x3 convolutions, 2x2x2 max pooling,
  skip connections at equal resolution, a final 1x1x1 convolution to the label
  channels, batch normalisation before each activation, and "doubling the number of
  channels already before max pooling" (the second convolution of a block is the one
  that widens).
* Code: a trimmed vendoring of wolny/pytorch-3dunet (MIT, (c) 2018 Adrian Wolny),
  pytorch3dunet/unet3d/{buildingblocks,model}.py -- create_conv, SingleConv,
  DoubleConv, Encoder, Decoder, create_encoders/decoders, the upsampling strategies
  and AbstractUNet->UNet3D.  Dropped: 2D variants, ResNet/squeeze-excitation blocks,
  config/trainer plumbing.
* Hyper-parameters: FLamby's Fed-IXI baseline (in_channels 1, out_classes 2,
  3 encoding blocks, 8 first-layer channels, batch normalisation, PReLU, linear
  upsampling, padding 1), which FLamby itself realises through fepegar/unet 0.7.7.

Two extensions to the vendored grammar are needed to express FLamby exactly:
  * 'p' = PReLU in the layer order, so FLamby's conv -> BatchNorm -> PReLU block is
    the order "cbp";
  * ``first_conv_order``: fepegar's first encoding block runs its FIRST convolution
    without normalisation (so that conv keeps its bias).  DoubleConv accepts a
    separate order for conv1 to reproduce that.

With f_maps = (16, 32, 64) the pytorch-3dunet channel rule (conv1 -> out//2, conv2 ->
out) produces exactly FLamby's widths:

    encoder 1     1 ->  8 -> 16          (conv1 has no BatchNorm, keeps its bias)
    encoder 2    16 -> 16 -> 32
    encoder 3    32 -> 32 -> 64          (the bottom block)
    decoder 1   (64 up + 32 skip) = 96 -> 32 -> 32
    decoder 2   (32 up + 16 skip) = 48 -> 16 -> 16
    classifier   16 -> 2, kernel 1, then softmax over the channel dimension

i.e. 246 156 parameters, identical to the FLamby model used by ../cfl_ixi.

Input volumes are 48x60x48.  Pooling floors odd sizes (60 -> 30 -> 15) and each
decoder interpolates back to its skip connection's exact size, so non-power-of-two
shapes are fine.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#: FLamby's Fed-IXI configuration, expressed in pytorch-3dunet terms.
FLAMBY_PRESET = dict(f_maps=16, num_levels=3, layer_order="cbp", upsample="trilinear",
                     first_block_no_norm=True)


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


# --------------------------------------------------------------- conv blocks
def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding,
                dropout_prob):
    """Ordered list of (name, module) for one convolution layer.

    order grammar: 'c' conv, 'b' BatchNorm3d, 'g' GroupNorm, 'r' ReLU, 'l' LeakyReLU,
    'e' ELU, 'p' PReLU (extension: FLamby's activation), 'd' Dropout3d.
    A convolution keeps its bias only when no normalisation accompanies it.
    """
    assert "c" in order, "conv layer MUST be present"
    assert order[0] not in "rlep", "non-linearity cannot be the first operation"
    modules = []
    for i, char in enumerate(order):
        if char == "r":
            modules.append(("ReLU", nn.ReLU(inplace=True)))
        elif char == "l":
            modules.append(("LeakyReLU", nn.LeakyReLU(inplace=True)))
        elif char == "e":
            modules.append(("ELU", nn.ELU(inplace=True)))
        elif char == "p":
            modules.append(("PReLU", nn.PReLU()))
        elif char == "c":
            bias = not ("g" in order or "b" in order)
            modules.append(("conv", nn.Conv3d(in_channels, out_channels, kernel_size,
                                              padding=padding, bias=bias)))
        elif char == "g":
            n_ch = in_channels if i < order.index("c") else out_channels
            g = 1 if n_ch < num_groups else num_groups
            assert n_ch % g == 0, "channels {} not divisible by groups {}".format(n_ch, g)
            modules.append(("groupnorm", nn.GroupNorm(num_groups=g, num_channels=n_ch)))
        elif char == "b":
            n_ch = in_channels if i < order.index("c") else out_channels
            modules.append(("batchnorm", nn.BatchNorm3d(n_ch)))
        elif char == "d":
            modules.append(("dropout", nn.Dropout3d(p=dropout_prob)))
        else:
            raise ValueError("unsupported layer type '{}' in order '{}'".format(char, order))
    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order="cbp",
                 num_groups=8, padding=1, dropout_prob=0.0):
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order,
                                        num_groups, padding, dropout_prob):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """The two convolutions of one U-Net level.

    encoder: in -> out//2 -> out  (never below ``in_channels``) -- the paper's
             "double the channels before pooling"
    decoder: in -> out -> out
    ``first_conv_order`` overrides the order of conv1 only (FLamby's first block runs
    it without normalisation).
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order="cbp",
                 num_groups=8, padding=1, upscale=2, dropout_prob=0.0,
                 first_conv_order=None):
        super().__init__()
        if encoder:
            conv1_in = in_channels
            conv1_out = out_channels if upscale == 1 else out_channels // 2
            if conv1_out < in_channels:
                conv1_out = in_channels
            conv2_in, conv2_out = conv1_out, out_channels
        else:
            conv1_in, conv1_out = in_channels, out_channels
            conv2_in, conv2_out = out_channels, out_channels
        self.add_module("SingleConv1",
                        SingleConv(conv1_in, conv1_out, kernel_size,
                                   first_conv_order or order, num_groups, padding,
                                   dropout_prob))
        self.add_module("SingleConv2",
                        SingleConv(conv2_in, conv2_out, kernel_size, order, num_groups,
                                   padding, dropout_prob))


# ------------------------------------------------------------- up/down paths
class Encoder(nn.Module):
    """[MaxPool3d] -> DoubleConv."""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, conv_layer_order="cbp", num_groups=8, padding=1,
                 upscale=2, dropout_prob=0.0, first_conv_order=None):
        super().__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size) if apply_pooling else None
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=True,
                                       kernel_size=conv_kernel_size,
                                       order=conv_layer_order, num_groups=num_groups,
                                       padding=padding, upscale=upscale,
                                       dropout_prob=dropout_prob,
                                       first_conv_order=first_conv_order)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        return self.basic_module(x)


class InterpolateUpsampling(nn.Module):
    """Parameter-free upsampling to the exact size of the skip connection.
    FLamby's ``upsampling_type="linear"`` -> trilinear in 3D."""

    def __init__(self, mode="trilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, encoder_features, x):
        size = encoder_features.size()[2:]
        if self.mode in ("trilinear", "linear"):
            return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        return F.interpolate(x, size=size, mode=self.mode)


class TransposeConvUpsampling(nn.Module):
    """Learned up-convolution (the 3D U-Net paper's 2x2x2 upconv), with an
    interpolation to the skip size when pooling floored an odd dimension."""

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                       stride=scale_factor, padding=1)

    def forward(self, encoder_features, x):
        size = encoder_features.size()[2:]
        x = self.conv(x)
        if x.size()[2:] != size:
            x = F.interpolate(x, size=size, mode="nearest")
        return x


class Decoder(nn.Module):
    """upsample -> concatenate the skip connection -> DoubleConv."""

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=2,
                 conv_layer_order="cbp", num_groups=8, padding=1, upsample="trilinear",
                 dropout_prob=0.0):
        super().__init__()
        if upsample == "deconv":
            self.upsampling = TransposeConvUpsampling(in_channels, out_channels,
                                                      conv_kernel_size, scale_factor)
        else:
            self.upsampling = InterpolateUpsampling(mode=upsample)
        self.basic_module = DoubleConv(in_channels, out_channels, encoder=False,
                                       kernel_size=conv_kernel_size,
                                       order=conv_layer_order, num_groups=num_groups,
                                       padding=padding, dropout_prob=dropout_prob)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = torch.cat((encoder_features, x), dim=1)
        return self.basic_module(x)


def create_encoders(in_channels, f_maps, conv_kernel_size, conv_padding, conv_upscale,
                    dropout_prob, layer_order, num_groups, pool_kernel_size,
                    first_block_no_norm=False):
    encoders = []
    for i, out_features in enumerate(f_maps):
        first_order = None
        if i == 0 and first_block_no_norm:
            # fepegar/FLamby: no normalisation on the very first convolution
            first_order = "".join(ch for ch in layer_order if ch not in "bg")
        encoders.append(Encoder(in_channels if i == 0 else f_maps[i - 1], out_features,
                                apply_pooling=(i != 0),   # no pooling in the first block
                                conv_layer_order=layer_order,
                                conv_kernel_size=conv_kernel_size, num_groups=num_groups,
                                pool_kernel_size=pool_kernel_size, padding=conv_padding,
                                upscale=conv_upscale, dropout_prob=dropout_prob,
                                first_conv_order=first_order))
    return nn.ModuleList(encoders)


def create_decoders(f_maps, conv_kernel_size, conv_padding, layer_order, num_groups,
                    upsample, dropout_prob):
    decoders = []
    rev = list(reversed(f_maps))
    for i in range(len(rev) - 1):
        # interpolation keeps the channel count, so the concat carries both levels;
        # a transposed convolution already maps down to the skip width
        in_features = rev[i] if upsample == "deconv" else rev[i] + rev[i + 1]
        decoders.append(Decoder(in_features, rev[i + 1], conv_kernel_size=conv_kernel_size,
                                conv_layer_order=layer_order, num_groups=num_groups,
                                padding=conv_padding, upsample=upsample,
                                dropout_prob=dropout_prob))
    return nn.ModuleList(decoders)


# -------------------------------------------------------------------- the net
class UNet3D(nn.Module):
    """3D U-Net returning per-voxel class probabilities (softmax over channels).

    Defaults are FLamby's Fed-IXI configuration: 3 levels of (16, 32, 64) block
    widths, conv -> BatchNorm -> PReLU, trilinear upsampling, 2 output classes.
    """

    def __init__(self, in_channels=1, out_channels=2, f_maps=16, num_levels=3,
                 layer_order="cbp", num_groups=8, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, conv_upscale=2, upsample="trilinear", dropout_prob=0.0,
                 first_block_no_norm=True, final_activation="softmax"):
        super().__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        assert len(f_maps) > 1, "at least 2 levels are required"
        self.f_maps = list(f_maps)
        self.encoders = create_encoders(in_channels, f_maps, conv_kernel_size, conv_padding,
                                        conv_upscale, dropout_prob, layer_order, num_groups,
                                        pool_kernel_size, first_block_no_norm)
        self.decoders = create_decoders(f_maps, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, upsample, dropout_prob)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        if final_activation == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None

    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.insert(0, x)          # deepest first
        encoder_features = encoder_features[1:]    # the bottom level is not a skip
        for decoder, features in zip(self.decoders, encoder_features):
            x = decoder(features, x)
        x = self.final_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


def build_model(cfg):
    """Instantiate the U-Net described by a Config: FLamby's preset plus overrides."""
    p = dict(FLAMBY_PRESET)
    if getattr(cfg, "f_maps", 0):
        p["f_maps"] = cfg.f_maps
    if getattr(cfg, "num_levels", 0):
        p["num_levels"] = cfg.num_levels
    if getattr(cfg, "layer_order", ""):
        p["layer_order"] = cfg.layer_order
    if getattr(cfg, "upsample", ""):
        p["upsample"] = cfg.upsample
    return UNet3D(in_channels=1, out_channels=2,
                  dropout_prob=getattr(cfg, "dropout_prob", 0.0), **p)


def model_factory(cfg):
    """-> a zero-argument callable, which is what Client/Server expect."""
    return lambda: build_model(cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":          # shape / parameter check
    net = UNet3D()
    out = net(torch.randn(2, 1, 48, 60, 48))
    print("FLamby Fed-IXI U-Net: f_maps={} params={:,} out={} sum_over_classes={:.3f}"
          .format(net.f_maps, count_parameters(net), tuple(out.shape),
                  float(out.sum(1).mean())))
    for name, p in net.named_parameters():
        if p.dim() == 5:
            print("  {:42s} {}".format(name, tuple(p.shape)))
