import torch


def gamma_EOTF(image, lumin):
    if image.shape[0] != lumin.shape[0]:
        raise ValueError('The number of images and luminance values must be the same')
    black = lumin / 1000
    return (lumin - black) * (image / 255) ** 2.2 + black


def sRGB_EOTF(image, lumin):
    if image.shape[0] != lumin.shape[0]:
        raise ValueError('The number of images and luminance values must be the same')
    black = lumin / 1000
    return (lumin - black) * sRGB2linear(image / 255) + black


def PQ_EOTF(image, lumin):
    if image.shape[0] != lumin.shape[0]:
        raise ValueError('The number of images and luminance values must be the same')
    black = lumin / 1000
    return PQ2linear(image / 255).clamp(0.005, lumin) + black


def sRGB2linear(image):
    result = torch.where(image > 0.04045, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
    return result


def PQ2linear(image):
    Lmax = 10000
    m1 = 2610 / 16384
    m2 = 2523 / 4096 * 128
    c1 = 3424 / 4096
    c2 = 2413 / 4096 * 32
    c3 = 2392 / 4096 * 32

    image_ = image ** (1 / m2)
    result = Lmax * torch.pow((image_ - c1).clamp(min=0) / (c2 - c3 * image_), 1 / m1)
    return result
