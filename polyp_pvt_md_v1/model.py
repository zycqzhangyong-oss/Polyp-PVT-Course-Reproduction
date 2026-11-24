"""Reuse the user's original Polyp-PVT layers; never overwrite lib/pvt.py."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from lib.pvt import PolypPVT
from md_block import MultiDomainBlock

VARIANTS = ('baseline', 'spatial', 'multidomain', 'spatial_frequency', 'spatial_wavelet')


class PolypPVTMultiDomain(PolypPVT):
    def __init__(self, variant, seed):
        # Initialize all original modules in exactly the baseline RNG order.
        super().__init__()
        del self.ca
        del self.sa
        branches = {'spatial': ('spatial',),
                    'multidomain': ('spatial', 'frequency', 'wavelet'),
                    'spatial_frequency': ('spatial', 'frequency'),
                    'spatial_wavelet': ('spatial', 'wavelet')}[variant]
        # Do not consume the training RNG when initializing the added module.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 100000)
            self.md_block = MultiDomainBlock(64, branches)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        cim_feature = self.md_block(x1)
        cfm_feature = self.CFM(self.Translayer4_1(x4), self.Translayer3_1(x3),
                               self.Translayer2_1(x2))
        guide = self.down05(self.Translayer2_0(cim_feature))
        sam_feature = self.SAM(cfm_feature, guide)
        p1 = F.interpolate(self.out_CFM(cfm_feature), scale_factor=8,
                           mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.out_SAM(sam_feature), scale_factor=8,
                           mode='bilinear', align_corners=False)
        return p1, p2


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_model(variant, seed):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant: ' + variant)
    torch.manual_seed(seed)
    model = PolypPVT() if variant == 'baseline' else PolypPVTMultiDomain(variant, seed)
    # Original constructor silently skips missing pretrained keys. Reject that.
    pretrained = torch.load('./pretrained_pth/pvt_v2_b2.pth', map_location='cpu')
    expected = model.backbone.state_dict()
    if not isinstance(pretrained, dict) or set(expected) - set(pretrained):
        raise ValueError('PVT pretrained file lacks backbone tensors. Use the original PVTv2-B2 state_dict.')
    for name, tensor in expected.items():
        candidate = pretrained[name]
        if (not isinstance(candidate, torch.Tensor) or candidate.shape != tensor.shape
                or not torch.isfinite(candidate).all().item()
                or not torch.equal(tensor.cpu(), candidate.cpu())):
            raise ValueError('PVT pretrained tensor mismatch: ' + name)
    common = {name: tensor for name, tensor in model.state_dict().items()
              if not name.startswith(('ca.', 'sa.', 'md_block.'))}
    metadata = {'common_initial_weights_sha256': tensor_fingerprint(common),
                'parameters': sum(p.numel() for p in model.parameters()),
                'replaced_module_parameters': sum(p.numel() for name, p in model.named_parameters()
                                                  if name.startswith(('ca.', 'sa.', 'md_block.')))}
    return model, metadata
