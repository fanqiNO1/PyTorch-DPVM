import os

import numpy as np
import torch


def pu2_encode(image):
    dir = os.path.dirname(__file__)
    l_lut = np.genfromtxt(f"{dir}/l_lut.csv", delimiter=",")
    P_lut = np.genfromtxt(f"{dir}/P_lut.csv", delimiter=",")

    l_min = -5.0
    l_max = 10.0
    pu_l = 31.9270
    pu_h = 149.9244
    N = 8192.0

    L = torch.log10(torch.maximum(torch.minimum(image, torch.tensor(10 ** l_max)), torch.tensor(10 ** l_min)))
    index = torch.floor((L - l_min) * N / (l_max - l_min)).type(torch.long)

    P_lut_t = torch.tensor(P_lut).to(image.device)

    encoded = (255.0 * (P_lut_t[index] - pu_l) / (pu_h - pu_l)).type(torch.float32)

    return encoded



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image = torch.arange(1e-5, 200, 10)
    encoded = pu2_encode(image)
    print(encoded)
    print(encoded / image)
    plt.plot(image.numpy(), encoded.numpy())
    plt.savefig("pu2_encode.png")
