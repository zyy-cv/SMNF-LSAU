import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import einops
import numpy as np

import models
from models import register
from utils import make_coord
import torch.nn.utils.spectral_norm as spectral_norm


@register('IDASR')
class IDASR(nn.Module):

    def __init__(self, encoder_spec, hyper_spec=None,grid_spec=None, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,is_cell=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.is_cell = is_cell
        self.r = 3
        self.r_area = (2 * self.r + 1) ** 2
        self.local_attn = True
        self.head = 8
        self.n_group_channels = 8
        self.dim = 256
        self.conv_num = 1
        self.imnet_num = 1
        self.offset_range_factor = 2
        kk = 5
        stride = 1
        pad_size = kk // 2 if kk != stride else 0

        self.encoder = models.make(encoder_spec)

        self.conv_ch = nn.Conv2d(self.encoder.out_dim, self.dim, kernel_size=3, padding=1)

        self.conv_vs = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1) for _ in range(self.conv_num)
        ])

        self.conv_qs = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            for _ in range(self.conv_num)
        ])

        self.conv_ks = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1)
            for _ in range(self.conv_num)
        ])

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kk, stride, pad_size, groups=8),
            LayerNormProxy(self.dim),
            nn.GELU(),
            nn.Conv2d(self.dim, 2*self.r_area, 1, 1, 0, bias=False)
        )

        self.imnet_coord = models.make(imnet_spec, args={'in_dim': self.r_area*2, 'out_dim': self.r_area})

        imnet_in_dim = self.dim * self.r_area + 2 if self.is_cell else self.dim * self.r_area

        self.imnets = nn.ModuleList([
            models.make(
                imnet_spec,
                args={'in_dim': imnet_in_dim}
            ) for _ in range(self.imnet_num)
        ])

    def gen_feat(self, inp,scale):

        self.conv_idx = 0

        self.feat = self.encoder(inp,scale)
        self.feat = self.conv_ch(self.feat)

        self.feat_q = self.conv_qs[self.conv_idx](self.feat)
        self.feat_k = self.conv_ks[self.conv_idx](self.feat)

        self.feat_v = self.conv_vs[self.conv_idx](self.feat)

        return self.feat

    def query_rgb(self, sample_coord, cell=None):
        self.prev_pred = None

        feat = self.feat

        device = feat.device

        bs, q_sample, _ = sample_coord.shape

        offsets = self.conv_offset(self.feat)

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
            unsqueeze(0).expand(bs, 2, *feat.shape[-2:])  ##lr Image 坐标 [bs, 2, h, w]

        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()  
        sample_coord_ = sample_coord_.unsqueeze(2)

        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)  #[bs, h*w ,1,2]

        offsets_sample = F.grid_sample(
            offsets, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)  # [bs, h*w ,1,256]

        rh = 2 / feat.shape[-2]  # h
        rw = 2 / feat.shape[-1]  # w

        r = self.r  # windows/2

        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh  # 7
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw  # 7
            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)  #[7*7,2]

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                self.feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1) #[bs, h*w ,1,256]
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)

            offsets_sample = offsets_sample.view(bs, -1, self.r_area, 2)

            if self.offset_range_factor >= 0:
                offset_range = torch.tensor([1.0 / (feat.shape[2] - 1.0), 1.0 / (feat.shape[3] - 1.0)], device=device).reshape(1, 1, 1, 2)
                offsets_sample = offsets_sample.tanh().mul(offset_range).mul(self.offset_range_factor)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta + offsets_sample

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                self.feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1) ##[bs, h*w, 49, 256]
            sample_feat_k = sample_feat_k.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            ).permute(0, 1, 3, 4, 2)

            sample_feat_v = F.grid_sample(
                self.feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1) ##[bs, h*w, 49, 256]
            sample_feat_v = sample_feat_v.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            )

        del offsets_sample, offset_range, sample_coord, offsets

        attn_value = (sample_feat_q @ sample_feat_k)
        attn_value = attn_value.reshape(
                bs, q_sample, self.head, self.r_area
            ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)


        rel_coord = sample_coord_ - sample_coord_k  #### [bs, h*w, 1, 2] - [bs, h*w, 49, 2]
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1] ## [bs, h*w, 49, 2]

        rel_cell = cell.clone() ## [bs,h*w, 2]
        rel_cell[..., 0] *= feat.shape[-2] ##[bs, hw, 2]
        rel_cell[..., 1] *= feat.shape[-1]

        coord = rel_coord
        weight_coord = self.imnet_coord(coord.contiguous().view(bs * q_sample, -1)).reshape(bs, q_sample, -1)  ##[bs, h*w, 1, 49]
        weight_coord = weight_coord.unsqueeze(3).expand(-1,-1,-1,self.head)

        attn = weight_coord+attn_value
        attn = F.softmax(attn, dim=-2)


        attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)

        sample_feat_v = torch.mul(sample_feat_v, attn) #[bs, h*w, 49, 256]

        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        del attn, weight_coord, attn_value, sample_feat_k, sample_coord_k, sample_feat_q, sample_feat_v

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets[0](feat_in)

        if self.prev_pred is None:
            self.prev_pred = pred
        else:
            pred = pred + self.prev_pred * 0.75
            self.prev_pred = pred

        pred = pred + F.grid_sample(self.inp, sample_coord_.flip(-1), mode='bilinear', \
                                    padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        return pred

    def forward(self, inp, coord, cell, scale):
        self.inp = inp
        self.gen_feat(inp,scale)
        return self.query_rgb(coord, cell)

    def make_coord(shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
            coord_x = -1+(2*i+1)/W
            coord_y = -1+(2*i+1)/H
            normalize to (-1, 1)
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)

        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def re_query_rgb(self, inp, sample_coord, cell=None):
        self.prev_pred = None

        feat = self.feat
        device = feat.device

        bs, q_sample, _ = sample_coord.shape

        offsets = self.conv_offset(self.feat)

        coord_lr = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1). \
            unsqueeze(0).expand(bs, 2, *feat.shape[-2:])

        # b, q, 1, 2
        sample_coord_ = sample_coord.clone()
        sample_coord_ = sample_coord_.unsqueeze(2)

        sample_coord_k = F.grid_sample(
            coord_lr, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)  # [bs, h*w ,1,2]

        offsets_sample = F.grid_sample(
            offsets, sample_coord_.flip(-1), mode='nearest', align_corners=False
        ).permute(0, 2, 3, 1)  # [bs, h*w ,1,256]

        rh = 2 / feat.shape[-2]  # h
        rw = 2 / feat.shape[-1]  # w

        r = self.r  # windows/2

        if self.local_attn:
            dh = torch.linspace(-r, r, 2 * r + 1).cuda() * rh  # 7
            dw = torch.linspace(-r, r, 2 * r + 1).cuda() * rw  # 7

            delta = torch.stack(torch.meshgrid(dh, dw), axis=-1).view(1, 1, -1, 2)  # [7*7,2]

            # Q - b, c, h, w -> b, c, q, 1 -> b, q, 1, c -> b, q, 1, h, c -> b, q, h, 1, c
            sample_feat_q = F.grid_sample(
                self.feat_q, sample_coord_.flip(-1), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)  # [bs, h*w ,1,256]
            sample_feat_q = sample_feat_q.reshape(
                bs, q_sample, 1, self.head, self.dim // self.head
            ).permute(0, 1, 3, 2, 4)

            offsets_sample = offsets_sample.view(bs, -1, self.r_area, 2)

            if self.offset_range_factor >= 0:
                offset_range = torch.tensor([1.0 / (feat.shape[2] - 1.0), 1.0 / (feat.shape[3] - 1.0)], device=device).reshape(1, 1, 1, 2)
                offsets_sample = offsets_sample.tanh().mul(offset_range).mul(self.offset_range_factor)

            # b, q, 1, 2 -> b, q, 49, 2
            sample_coord_k = sample_coord_k + delta + offsets_sample

            # K - b, c, h, w -> b, c, q, 49 -> b, q, 49, c -> b, q, 49, h, c -> b, q, h, c, 49
            sample_feat_k = F.grid_sample(
                self.feat_k, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1)  ##[bs, h*w, 49, 256]
            sample_feat_k = sample_feat_k.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            ).permute(0, 1, 3, 4, 2)

            sample_feat_v = F.grid_sample(
                self.feat_v, sample_coord_k.flip(-1), mode='nearest', align_corners=False
            ).permute(0, 2, 3, 1)  ##[bs, h*w, 49, 256]
            sample_feat_v = sample_feat_v.reshape(
                bs, q_sample, self.r_area, self.head, self.dim // self.head
            )


        rel_coord = sample_coord_ - sample_coord_k  #### [bs, h*w, 1, 2] - [bs, h*w, 49, 2]
        rel_coord[..., 0] *= feat.shape[-2]
        rel_coord[..., 1] *= feat.shape[-1]  ## [bs, h*w, 49, 2]

        rel_cell = cell.clone()  ## [bs,h*w, 2]
        # rel_cell = rel_cell.unsqueeze(1).repeat(1, q_sample, 1)
        rel_cell[..., 0] *= feat.shape[-2]  ##[bs, hw, 2]
        rel_cell[..., 1] *= feat.shape[-1]

        attn_value = (sample_feat_q @ sample_feat_k)
        attn_value = attn_value.reshape(
            bs, q_sample, self.head, self.r_area
        ).permute(0, 1, 3, 2) / np.sqrt(self.dim // self.head)

        coord = rel_coord
        weight_coord = self.imnet_coord(coord.contiguous().view(bs * q_sample, -1)).reshape(bs, q_sample,
                                                                                            -1)  ##[bs, h*w, 1, 49]
        weight_coord = weight_coord.unsqueeze(3).expand(-1, -1, -1, self.head)

        attn = weight_coord + attn_value
        attn = F.softmax(attn, dim=-2)

        attn = attn.reshape(bs, q_sample, self.r_area, self.head, 1)

        sample_feat_v = torch.mul(sample_feat_v, attn)  # [bs, h*w, 49, 256]

        feat_in = sample_feat_v.reshape(bs, q_sample, -1)

        if self.is_cell:
            feat_in = torch.cat([feat_in, rel_cell], dim=-1)

        pred = self.imnets[0](feat_in)

        if self.prev_pred is None:
            self.prev_pred = pred
        else:
            pred = pred + self.prev_pred * 0.75
            self.prev_pred = pred

        pred = pred + F.grid_sample(inp, sample_coord_.flip(-1), mode='bilinear', \
                                    padding_mode='border', align_corners=False)[:, :, :, 0].permute(0, 2, 1)

        return pred


class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

def main():
    data = torch.rand((256,256,3))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='C:/Users/liu/Desktop/tt/liif-main/configs/train-div2k/train_edsr-baseline-liif.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    model_spec = config['model']
    model_args = model_spec['args']
    model = IDASR(**model_args)
    output = model(data)

if __name__=='__main__':
    main()