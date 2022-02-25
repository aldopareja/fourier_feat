import pywt
import torch
from torch.autograd import Variable


def create_filters(device, wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    dec_hi = torch.Tensor(w.dec_hi[::-1]).to(device)
    dec_lo = torch.Tensor(w.dec_lo[::-1]).to(device)

    filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                           dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                           dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                           dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    return filters


@torch.no_grad()
def wt(vimg, filters):
    # (Input is N, C, H, W)
    B, _, H, W = vimg.shape
    vimg = vimg.permute(1, 0, 2, 3).reshape(-1, 1, H, W)
    padded = torch.nn.functional.pad(vimg, (2, 2, 2, 2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:, None]), stride=2)
    res = res.view(-1, 2, H // 2, W // 2).transpose(1, 2).contiguous().view(-1, B, H, W)
    # (Returns N, C, H, W)
    return res.permute(1, 0, 2, 3)
