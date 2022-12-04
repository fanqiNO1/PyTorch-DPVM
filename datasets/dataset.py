import random

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from .utils import get_image, random_flip, random_rotate, to_tensor


class LocVisVC(Dataset):
    """
    - root/
        - data/
            - dataset1.csv
            - dataset2.csv
            - ...
        - likelihood
            - dataset1.csv
            - dataset2.csv
            - ...
        - reference
            - dataset1
                - image1
                    - 0.png
                    - 1.png
                    - ...
                - ...
            - ...
        - distortion
            - dataset1
                - image1
                    - 0.png
                    - 1.png
                    - ...
                - ...
            - ...
        - label
            - dataset1
                - image1
                    - 0.png
                    - 1.png
                    - ...
                - ...
            - ...
        - train.txt
        - test.txt
    """
    def __init__(self, path, is_training=True, seed=0x66ccff):
        self.path = path
        self.is_training = is_training
        self.seed = seed
        random.seed(self.seed)
        self.files = []
        self.file = "train.txt" if is_training else "test.txt"
        with open(f"{path}/{self.file}", "r") as f:
            self.files = f.readlines()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # dataset/image/patch
        file = self.files[index].strip()
        dataset, image, patch = file.split("/")
        reference = get_image(f"{self.path}/reference/{dataset}/{image}/{patch}")
        distortion = get_image(f"{self.path}/distortion/{dataset}/{image}/{patch}")
        label = get_image(f"{self.path}/label/{dataset}/{image}/{patch}", is_gray=True)
        # data augmentation
        if self.is_training:
            flip_mode = random.randint(1, 4)
            rotate_mode = random.randint(1, 4)
            reference = random_flip(reference, flip_mode)
            reference = random_rotate(reference, rotate_mode)
            distortion = random_flip(distortion, flip_mode)
            distortion = random_rotate(distortion, rotate_mode)
            label = random_flip(label, flip_mode)
            label = random_rotate(label, rotate_mode)
        # to tensor
        reference = to_tensor(reference)
        distortion = to_tensor(distortion)
        label = to_tensor(label)
        # lumin, ppd, n
        data = pd.read_csv(f"{self.path}/data/{dataset}.csv")
        lumin = torch.tensor([data[data["base_fname"] == image]["peak_lum"].values[0]], dtype=torch.float32)
        ppd = torch.tensor([data[data["base_fname"] == image]["ppd"].values[0]], dtype=torch.float32)
        n = torch.tensor([data[data["base_fname"] == image]["n"].values[0]], dtype=torch.float32)
        # likelihood
        likelihood = self.__load_likelihood(dataset)
        return {
            "reference": reference,
            "distortion": distortion,
            "label": label,
            "lumin": lumin.reshape(-1, 1, 1),
            "ppd": ppd.reshape(-1, 1, 1),
            "n": n.reshape(-1, 1, 1),
            "likelihood": likelihood,
        }


    def __load_likelihood(self, dataset):
        likelihood = torch.zeros((32, 32))
        data = np.loadtxt(f"{self.path}/likelihood/{dataset}.csv", delimiter=",", skiprows=1)
        likelihood[:data.shape[0], :data.shape[1]] = torch.tensor(data)
        return likelihood


if __name__ == "__main__":
    train_dataset = LocVisVC("dataset", is_training=True)
    test_dataset = LocVisVC("dataset", is_training=False)
    print("train_dataset:", len(train_dataset))
    print("test_dataset:", len(test_dataset))
    print("reference:", train_dataset[0]["reference"].shape, train_dataset[0]["reference"].max(), train_dataset[0]["reference"].min())
    print("distortion:", train_dataset[0]["distortion"].shape, train_dataset[0]["distortion"].max(), train_dataset[0]["distortion"].min())
    print("label:", train_dataset[0]["label"].shape, train_dataset[0]["label"].max(), train_dataset[0]["label"].min())
    print("lumin:", train_dataset[0]["lumin"].shape, train_dataset[0]["lumin"])
    print("ppd:", train_dataset[0]["ppd"].shape, train_dataset[0]["ppd"])
    print("n:", train_dataset[0]["n"].shape, train_dataset[0]["n"])
    print("likelihood:", train_dataset[0]["likelihood"].shape)
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, num_workers=0)
    for i, data in enumerate(test_dataloader):
        print("reference dataloader", data["reference"].shape)
        print("distortion dataloader", data["distortion"].shape)
        print("label dataloader", data["label"].shape)
        print("lumin dataloader", data["lumin"].shape)
        print("ppd dataloader", data["ppd"].shape)
        print("n dataloader", data["n"].shape)
        print("likelihood dataloader", data["likelihood"].shape)
        break