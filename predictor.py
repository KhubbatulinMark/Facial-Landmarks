"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import pickle
import sys
import time
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import tqdm
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn import functional as fnn
from torch.utils import data
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from model import create_model

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 256
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        # self.size = torch.tensor(size, dtype=torch.float)
        self.size = np.asarray(size, dtype=np.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train"):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks_new.csv') if split != "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp), total=num_lines + 1):
                if i == 0:
                    continue  # skip header
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                if split != 'test':
                    continue
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')


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


def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", "-c", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=128, type=int)  # 512 is OK for resnet18 finetuning @ 3GB of VRAM
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def main(args):
    os.makedirs("runs", exist_ok=True)

    # 1. prepare data & models
    train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.39963884, 0.31994772, 0.28253724],
                                             std=[0.33419772, 0.2864468, 0.26987]), ("image",)
                        ),
    ])

    print("Reading data...")
    train_dataset = ThousandLandmarksDataset(os.path.join(args.data, "train"), train_transforms, split="train")
    print(f"Train sample size {len(train_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                  shuffle=False, drop_last=False)

    test_dataset = ThousandLandmarksDataset(os.path.join(args.data, "test"), train_transforms, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                 shuffle=False, drop_last=False)

    device = torch.device("cuda:0") if args.gpu else torch.device("cpu")

    print("Creating model...")
    model = create_model()
    model.to(device)

    with open(os.path.join("runs", f"{args.checkpoint}_best.pth"), "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    train_predictions = predict(model, train_dataloader, device)
    with open(os.path.join("runs", f"{args.checkpoint}_train_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": train_dataset.image_names,
                     "landmarks": train_predictions}, fp)

    test_predictions = predict(model, test_dataloader, device)
    with open(os.path.join("runs", f"{args.checkpoint}_test_predictions.pkl"), "wb") as fp:
        pickle.dump({"image_names": test_dataset.image_names,
                     "landmarks": test_predictions}, fp)

    create_submission(args.data, test_predictions, os.path.join("runs", f"{args.checkpoint}_submit.csv"))


if __name__ == "__main__":
    args = parse_arguments()
    sys.exit(main(args))
