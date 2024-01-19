import torch
import torch.nn.functional as F
import random

def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute padding tuple."""
    # from kornia lib
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def hr2lr(img, kernel, noise=None, factor=2.0):
    
    
    if noise:
        noise_mean, noise_std = noise
        img_noised = img + torch.normal(noise_mean, noise_std, size=img.shape).to(img.device)
    else:
        img_noised = img

    b, c, h, w = img.shape

    kernel = kernel.expand(1,c,-1,-1)
    hk, wk = kernel.shape[-2:]
    kernel = kernel.reshape(-1, 1, hk, wk)

    padding_shape: list[int] = _compute_padding([hk, wk])

    img_pad = F.pad(img_noised, padding_shape, mode='reflect')
    img_pad = img_pad.view(-1, kernel.size(0), img_pad.size(-2), img_pad.size(-1))

    img_blur = F.conv2d(img_pad, kernel, groups=kernel.size(0), padding=0, stride=1)
    img_blur = img_blur.view(b, c, h, w)

    #mode = random.choice(['bilinear', 'bicubic'])
    mode = 'bilinear'
    out = F.interpolate(
        img_blur,
        size=(int(h//factor), int(w//factor)),
        mode=mode,
        align_corners=False,
    )
    
    return out