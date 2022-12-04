import argparse
import os
import random
import shutil

import cv2
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="LocVisVC", help="input path")
    parser.add_argument("--output_path", type=str, default="dataset", help="output path")
    parser.add_argument("--base_ppd", type=int, default=60, help="base pixel per degree")
    parser.add_argument("--patch_size", type=int, default=48, help="patch size")
    parser.add_argument("--stride", type=int, default=100, help="stride")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="train/test split ratio")
    parser.add_argument("--seed", type=int, default=0x66ccff, help="random seed")
    args = parser.parse_args()
    return args


def process_single_dataset(input_path, output_path, base_ppd=60, patch_size=48, stride=100):
    data = f"{input_path}/marking_stimuli.csv"
    data = pd.read_csv(data)
    dataset_name = input_path.split("/")[-1]
    # scene,level,base_fname,dataset,n,peak_lum,ppd
    fnames = data["base_fname"].values
    input_types = ["reference", "test", "marking"]
    output_types = ["reference", "distortion", "label"]
    for fname in tqdm(fnames):
        patches = dict()
        for output_type in output_types:
            patches[output_type] = []
        for i in range(len(input_types)):
            image_path = f"{input_path}/{fname}-{input_types[i]}.png"
            image = cv2.imread(image_path)
            # pixel per degree
            ppd = data[data["base_fname"] == fname]["ppd"].values[0]
            scale = base_ppd / ppd
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            # generate patches
            patch_path = f"{output_path}/{output_types[i]}/{dataset_name}/{fname}"
            if not os.path.exists(patch_path):
                os.makedirs(patch_path)
            width, height = image.shape[:2]
            patch_index = 0
            for j in range(0, width, stride):
                for k in range(0, height, stride):
                    if j + patch_size > width or k + patch_size > height:
                        continue
                    patch = image[j:j + patch_size, k:k + patch_size]
                    if patch.shape[:2] != (patch_size, patch_size):
                        continue
                    patches[output_types[i]].append(patch)
        # save patches
        patch_index = 0
        for i in range(len(patches["reference"])):
            reference = patches["reference"][i]
            distortion = patches["distortion"][i]
            label = patches["label"][i]
            if np.mean(reference - distortion) < 1e-6:
                continue
            else:
                cv2.imwrite(f"{output_path}/reference/{dataset_name}/{fname}/{patch_index}.png", reference)
                cv2.imwrite(f"{output_path}/distortion/{dataset_name}/{fname}/{patch_index}.png", distortion)
                cv2.imwrite(f"{output_path}/label/{dataset_name}/{fname}/{patch_index}.png", label)
                patch_index += 1
        # logging.info(f"There should be {(j / stride + 1) * (k / stride + 1)} patches, but there are {patch_index} patches.")
    shutil.copy(f"{input_path}/marking_stimuli.csv", f"{output_path}/data/{dataset_name}.csv")
    shutil.copy(f"{input_path}/likelihood.csv", f"{output_path}/likelihood/{dataset_name}.csv")
    


def process_LocVisVC(input_path, output_path, base_ppd=60, patch_size=48, stride=100):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dirs = ["reference", "distortion", "label", "data", "likelihood"]
    for dir in dirs:
        if not os.path.exists(f"{output_path}/{dir}"):
            os.makedirs(f"{output_path}/{dir}")
    datasets = os.listdir(input_path)
    for dataset in datasets:
        if os.path.isdir(f"{input_path}/{dataset}"):
            logging.info(f"Processing {dataset}...")
            process_single_dataset(f"{input_path}/{dataset}", output_path, base_ppd, patch_size, stride)


def train_test_split(input_path, ratio=0.9, seed=0x66ccff):
    logging.info("Generating train/test split...")
    files = []
    random.seed(seed)
    datasets = os.listdir(f"{input_path}/reference")
    for dataset in datasets:
        images = os.listdir(f"{input_path}/reference/{dataset}")
        for image in images:
            patches = os.listdir(f"{input_path}/reference/{dataset}/{image}")
            for patch in patches:
                files.append(f"{dataset}/{image}/{patch}")
    random.shuffle(files)
    logging.info(f"Total number of patches: {len(files)}")
    train_size = int(len(files) * ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    with open(f"{input_path}/train.txt", "w") as f:
        for file in train_files:
            f.write(f"{file}\n")
    with open(f"{input_path}/test.txt", "w") as f:
        for file in test_files:
            f.write(f"{file}\n")


def main(args):
    process_LocVisVC(args.input_path, args.output_path, args.base_ppd, args.patch_size, args.stride)
    train_test_split(args.output_path, args.split_ratio, args.seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    main(args)
