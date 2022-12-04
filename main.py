import os

import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

from datasets.dataset import LocVisVC
from models.alexnet import SiameseAlexNet
from transforms.display_model import gamma_EOTF
from transforms.pu2_encode import pu2_encode


def rgb2gray(image):
    # shape (b, 3, h, w)
    result = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    return result.unsqueeze(1)


def PSNR(pred, label, max_val=255):
    mse = torch.mean((pred - label) ** 2)
    return 10 * torch.log10(max_val ** 2 / mse)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    args = parser.parse_args()
    return args


def train(model, dataloader, criterion, optimizer, device, log_interval, epoch):
    model.train()
    for batch_index, data in enumerate(dataloader):
        optimizer.zero_grad()
        reference = data["reference"].to(device)
        distortion = data["distortion"].to(device)
        label = data["label"].to(device)
        lumin = data["lumin"].to(device)
        ppd = data["ppd"].to(device)
        n = data["n"].to(device)
        likelihood = data["likelihood"].to(device)

        reference = gamma_EOTF(reference, lumin)
        distortion = gamma_EOTF(distortion, lumin)
        ref_lumin = rgb2gray(reference)
        dis_lumin = rgb2gray(distortion)
        ref_ratio = pu2_encode(ref_lumin) / (ref_lumin + 1e-8)
        dis_ratio = pu2_encode(dis_lumin) / (dis_lumin + 1e-8)
        reference = reference * ref_ratio
        distortion = distortion * dis_ratio

        pred = model(reference, distortion, ppd)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        if (batch_index + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_index + 1}, Loss {loss.item()}, PSNR {PSNR(pred, label).item()}, SSIM {ssim(pred, label).item()}")


def test(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for batch_index, data in enumerate(dataloader):
            reference = data["reference"].to(device)
            distortion = data["distortion"].to(device)
            label = data["label"].to(device)
            lumin = data["lumin"].to(device)
            ppd = data["ppd"].to(device)
            n = data["n"].to(device)
            likelihood = data["likelihood"].to(device)

            reference = gamma_EOTF(reference, lumin)
            distortion = gamma_EOTF(distortion, lumin)
            ref_lumin = rgb2gray(reference)
            dis_lumin = rgb2gray(distortion)
            ref_ratio = pu2_encode(ref_lumin) / (ref_lumin + 1e-8)
            dis_ratio = pu2_encode(dis_lumin) / (dis_lumin + 1e-8)
            reference = reference * ref_ratio
            distortion = distortion * dis_ratio
            pred = model(reference, distortion, ppd)
            loss = criterion(pred, label)

            total_loss += loss.item()
            total_psnr += PSNR(pred, label).item()
            total_ssim += ssim(pred, label).item()
        print(f"Epoch {epoch + 1}, Test Loss {total_loss / (batch_index + 1)} PSNR {total_psnr / (batch_index + 1)} SSIM {total_ssim / (batch_index + 1)}")


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = f"cuda:{args.device}" if args.device != -1 else "cpu"
    train_dataset = LocVisVC("dataset", is_training=True)
    test_dataset = LocVisVC("dataset", is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = SiameseAlexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, device, args.log_interval, epoch)
        test(model, test_loader, criterion, device, epoch)
        scheduler.step()
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{epoch + 1}.pth"))


if __name__ == "__main__":
    main()
