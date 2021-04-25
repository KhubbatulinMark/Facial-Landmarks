"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import sys
from argparse import ArgumentParser

import numpy as np
import tqdm

import cv2
import torch
import torch.optim as optim
from torch.nn import functional as fnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A

from model import create_model
from utils import NUM_PTS, CROP_SIZE
from utils import ScaleMinSideToSize, CropCenter, TransformByKeys, RandomApply, RandomRotate, RandomPadAndResize, LandmarksAugmentation
from utils import ThousandLandmarksDataset
from utils import restore_landmarks_batch


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=128, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--epochs", "-e", default=15, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    print(f"training... {len(loader)} iters \n")
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss, real_val_loss = [], []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

        # Расчет "правильного" лосса
        fs = batch["scale_coef"].numpy()
        # Вытаскиваем инфо о кромках
        margins_x = batch["crop_margin_x"].numpy()
        margins_y = batch["crop_margin_y"].numpy()
        # Пересчитываем в исходные координаты предсказания модели
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)
        # Пересчитываем в исходные координаты ground_true - координаты
        landmarks = landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))
        real_landmarks = restore_landmarks_batch(landmarks, fs, margins_x, margins_y)
        # Добавяем MSE в список real_val_loss
        real_loss = (prediction.reshape(-1) - real_landmarks.reshape(-1)) ** 2
        real_val_loss.append(np.mean(real_loss))

    return np.mean(val_loss), np.mean(real_val_loss)


def predict(model, loader, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), NUM_PTS, 2))
    for i, batch in enumerate(tqdm.tqdm(loader, total=len(loader), desc="test prediction...")):
        images = batch["image"].to(device)

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

        fs = batch["scale_coef"].numpy()  # B
        margins_x = batch["crop_margin_x"].numpy()  # B
        margins_y = batch["crop_margin_y"].numpy()  # B
        prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
        predictions[i * loader.batch_size: (i + 1) * loader.batch_size] = prediction

    return predictions


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = transforms.Compose([
        # RandomHorizontalFlip(p=0.5),
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        # RandomApply([
        #     RandomPadAndResize(percent=0.15),
        #     RandomRotate(max_angle=15),
        # ], p=[0.15, 0.85]),
        LandmarksAugmentation,
        # TransformByKeys(transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.03), ("image",)),
        # TransformByKeys(transforms.RandomGrayscale(p=0.1), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ("image",)),
    ])


    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    print(f"Train sample size {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=True, drop_last=True)
    val_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                shuffle=False, drop_last=False)
    print(f"Validation sample size {len(val_dataset)}")
    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    print("Creating model...")
    model = create_model()

    model.to(device)

    optimizer = optim.AdamW(model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-03,
        amsgrad=True)

    loss_fn = fnn.mse_loss

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=args.epochs
    )
    # 2. train & validate
    print("Ready for training...")
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        train_loss = train(model,
                           train_dataloader,
                           loss_fn, optimizer,
                           device=device,
                           scheduler=scheduler
                           )

        val_loss, real_val_loss = validate(model,
                            val_dataloader,
                            loss_fn,
                            device=device,
                            )

        print("Epoch #{:2}:\ttrain loss: {:5.3}\tval loss: {:5.3} /{:5.3}".format(epoch, train_loss, val_loss,
                                                                                  real_val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(os.path.join("runs", f"{args.name}_best.pth"), "wb") as fp:
                torch.save(model.state_dict(), fp)


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
