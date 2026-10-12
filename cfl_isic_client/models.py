"""The Fed-ISIC2019 baseline model: EfficientNet-B0.

Lineage
-------
* Architecture: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional
  Neural Networks", arXiv:1905.11946 (ICML 2019).  B0 is the compound-scaling base
  network found by neural architecture search: a stem convolution, 16 inverted
  residual MBConv blocks with squeeze-and-excitation (kernel 3 or 5, expansion 1 or
  6, 5 resolution stages), a 1x1 head to 1280 channels, global average pooling and a
  linear classifier.  The reference implementation is TensorFlow
  (github.com/tensorflow/tpu/tree/master/models/official/efficientnet).
* Code: FLamby does not re-implement it either -- flamby/datasets/fed_isic2019/model.py
  imports the PyTorch port by Luke Melas-Kyriazi (``efficientnet_pytorch``), loads the
  ImageNet weights and swaps the classifier:

      self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
      nftrs = self.base_model._fc.in_features          # 1280
      self.base_model._fc = nn.Linear(nftrs, 8)        # 8 ISIC classes

  That is exactly what ``Baseline`` below does, so the network is byte-for-byte the
  one FLamby trains: 4 017 796 parameters (4.01 M of EfficientNet-B0 minus its
  1000-class head, plus the 8-class head).

Inputs are 200x200 crops normalised with the ImageNet statistics (see data_utils).
Outputs are raw logits; the weighted focal loss and the balanced-accuracy metric in
metrics.py consume them directly, as in FLamby.
"""
import contextlib
import io

import torch
import torch.nn as nn


class Baseline(nn.Module):
    """FLamby's fed_isic2019 Baseline: ImageNet EfficientNet-B0 with an 8-class head."""

    def __init__(self, pretrained=True, arch_name="efficientnet-b0", num_classes=8):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        with contextlib.redirect_stdout(io.StringIO()):   # silence "Loaded pretrained..."
            self.base_model = (EfficientNet.from_pretrained(arch_name) if pretrained
                               else EfficientNet.from_name(arch_name))
        nftrs = self.base_model._fc.in_features           # 1280 for B0
        self.base_model._fc = nn.Linear(nftrs, num_classes)

    def forward(self, image):
        return self.base_model(image)


def build_model(cfg):
    return Baseline(pretrained=getattr(cfg, "pretrained", True))


def model_factory(cfg):
    """-> a zero-argument callable, which is what Client/Server expect."""
    return lambda: build_model(cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = Baseline(pretrained=False)
    out = net(torch.randn(2, 3, 200, 200))
    print("EfficientNet-B0 + 8-class head: params={:,} out={}".format(
        count_parameters(net), tuple(out.shape)))
