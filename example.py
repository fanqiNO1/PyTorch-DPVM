import argparse
import cv2
import numpy as np
import torch

from datasets.utils import get_image, to_tensor
from models.alexnet import SiameseAlexNet
from transforms.display_model import gamma_EOTF
from transforms.pu2_encode import pu2_encode


def parse_args():
    parser = argparse.ArgumentParser()
    # For images
    parser.add_argument("--ref", type=str, default="reference.png")
    parser.add_argument("--dist", type=str, default="distortion.png")
    parser.add_argument("--ppd", type=int, default=60)
    parser.add_argument("--lumin", type=float, default=110)
    # For model
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--load_path", type=str, default="models/alexnet.pth")
    parser.add_argument("--device", type=int, default=0)
    # For output
    parser.add_argument("--merging_method", type=str, default="mean")
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()
    return args


def rgb2gray(image):
    # shape (b, 3, h, w)
    result = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    return result.unsqueeze(1)


def get_input(image, patch_size=48):
    width, height, channels = image.shape
    temp = np.concatenate((np.flip(image, 0), image, np.flip(image, 0)), 0)
    result = np.concatenate((np.flip(temp, 1), temp, np.flip(temp, 1)), 1)
    result = result[width - patch_size: 2 * width + patch_size, height - patch_size: 2 * height + patch_size, :]
    return result



def main(args):
    ratio = 60 / args.ppd
    patch_size = 48
    window_step = 8
    overlap_number = int(patch_size / window_step)
    reference = get_image(args.ref)
    reference = cv2.resize(reference, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    distortion = get_image(args.dist)
    distortion = cv2.resize(distortion, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    origin_width, origin_height, _ = reference.shape
    reference = get_input(reference)
    distortion = get_input(distortion)
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    reference = torch.tensor(to_tensor(reference)).unsqueeze(0)
    distortion = torch.tensor(to_tensor(distortion)).unsqueeze(0)
    lumin = torch.tensor([args.lumin]).reshape(-1, 1, 1, 1)
    reference = gamma_EOTF(reference, lumin)
    distortion = gamma_EOTF(distortion, lumin)
    ref_lumin = rgb2gray(reference)
    dis_lumin = rgb2gray(distortion)
    ref_ratio = pu2_encode(ref_lumin) / (ref_lumin + 1e-8)
    dis_ratio = pu2_encode(dis_lumin) / (dis_lumin + 1e-8)
    reference = reference * ref_ratio
    distortion = distortion * dis_ratio

    model = SiameseAlexNet()
    model.load_state_dict(torch.load(args.load_path, map_location="cpu"))
    model.to(device)
    model.eval()

    ref_patches = []
    dis_patches = []
    ppd_patches = []
    x_i = []
    y_i = []
    # BCHW
    h, w = reference.shape[2:]
    for i in range(0, h - patch_size, window_step):
        for j in range(0, w - patch_size, window_step):
            ref_patches.append(reference[:, :, i: i + patch_size, j: j + patch_size])
            dis_patches.append(distortion[:, :, i: i + patch_size, j: j + patch_size])
            ppd_patches.append(args.ppd)
            x_i.append(i)
            y_i.append(j)
    ref_patches = torch.cat(ref_patches, 0)
    dis_patches = torch.cat(dis_patches, 0)
    ppd_patches = torch.tensor(ppd_patches).reshape(-1, 1, 1, 1).to(device)

    ref_inputs = torch.split(ref_patches, args.batch_size, 0)
    dis_inputs = torch.split(dis_patches, args.batch_size, 0)
    ppd_inputs = torch.split(ppd_patches, args.batch_size, 0)
    result = np.ones((h, w, overlap_number * overlap_number))
    with torch.no_grad():
        for i in range(len(ref_inputs)):
            ref_input = ref_inputs[i].to(device)
            dis_input = dis_inputs[i].to(device)
            ppd_input = ppd_inputs[i].to(device)
            output = model(ref_input, dis_input, ppd_input).cpu()
            for j in range(output.shape[0]):
                xx = x_i[i * args.batch_size + j]
                yy = y_i[i * args.batch_size + j]
                layer_index = int((xx % patch_size)/ window_step) * overlap_number + int((yy % patch_size) / window_step)
                result[xx: xx + patch_size, yy: yy + patch_size, layer_index] = output[j, 0, :, :].numpy()
    

    if args.merging_method == "mean":
        result = np.mean(result, axis=2)
    elif args.merging_method == "median":
        result = np.median(result, axis=2)
    elif args.merging_method == "max":
        result = np.max(result, axis=2)
    elif args.merging_method == "percentile":
        result = np.percentile(result, 95, axis=2)
    else:
        raise NotImplementedError

    result = result[patch_size: patch_size + origin_width, patch_size: patch_size + origin_height]
    result = cv2.resize(result, (0, 0), fx=1 / ratio, fy=1 / ratio, interpolation=cv2.INTER_AREA)
    # result = cv2.applyColorMap(result.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(args.output, result)
    


if __name__ == "__main__":
    args = parse_args()
    main(args)