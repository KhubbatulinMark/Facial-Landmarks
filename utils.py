import os
import pickle

import random
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
import torchvision.transforms.functional as TF
from PIL import Image
import albumentations as A

from torch.utils import data


np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 256
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
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

class Identity(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class RandomApply(object):
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, sample):
        ind = np.random.choice(len(self.transforms), p=self.p)
        transform = self.transforms[ind]
        sample = transform(sample)
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def revert_landmarks(labels):
        new_labels = np.concatenate((
            labels[128:256],
            labels[0:128],
            np.flipud(labels[256:546].reshape(-1, 2)).reshape(-1),
            labels[674:802],
            labels[546:674],
            labels[928:1054],
            labels[802:928],
            labels[1054:1174],
            labels[1428:1682],
            labels[1174:1428],
            np.flipud(labels[1682:1746].reshape(-1, 2)).reshape(-1),
            np.flipud(labels[1746:1810].reshape(-1, 2)).reshape(-1),
            np.flipud(labels[1810:1874].reshape(-1, 2)).reshape(-1),
            np.flipud(labels[1874:1938].reshape(-1, 2)).reshape(-1),
            labels[1940:1942],
            labels[1938:1940]
        ))
        return torch.from_numpy(new_labels)

    def flip_landmarks(self, labels, w):
        landmarks = labels.reshape(-1, 2)
        landmarks[:, 0] = w - landmarks[:, 0]
        labels = landmarks.reshape(-1)
        labels = self.revert_landmarks(labels)
        return labels

    def __call__(self, sample):
        if random.random() < self.p:
            _, w, _ = sample['image'].shape
            sample['image'] = cv2.flip(sample['image'], 1)
            sample['landmarks'] = self.flip_landmarks(sample['landmarks'], w)
        return sample


class RandomPadAndResize(object):
    def __init__(self, percent=0.15):
        self.percent = percent

    def __call__(self, sample):
        w, h = sample['image'].size
        left_pad = int(random.random() * self.percent * w)
        right_pad = int(random.random() * self.percent * w)
        top_pad = int(random.random() * self.percent * h)
        bottom_pad = int(random.random() * self.percent * h)

        sample['image'] = TF.pad(sample['image'], padding=(left_pad, top_pad, right_pad, bottom_pad))

        landmarks = sample['landmarks'].reshape(-1, 2)
        landmarks[:, 0] += left_pad
        landmarks[:, 1] += top_pad

        new_w, new_h = sample['image'].size
        fw = w / new_w
        fh = h / new_h
        sample['image'] = TF.resize(sample['image'], (w, h))

        landmarks[:, 0] = landmarks[:, 0] * fw
        landmarks[:, 1] = landmarks[:, 1] * fh
        sample['landmarks'] = landmarks.reshape(-1)
        return sample


class DropoutAugmentor(object):
    def __init__(self, p=(0., 0.01), size=(CROP_SIZE, CROP_SIZE)):
        self.p = p
        self.size = size
        self.px_cnt = self.size[0] * self.size[1]

    def random_coordinates(self, n):
        salt_n = random.randint(0, n)
        pepper_n = n - salt_n
        salt_coords = np.random.randint(0, self.size[0], (salt_n, 2))
        pepper_coords = np.random.randint(0, self.size[0], (pepper_n, 2))
        return salt_coords, pepper_coords

    def __call__(self, sample):
        n = int((random.random() * (self.p[1] - self.p[0]) + self.p[0]) * self.px_cnt)
        salt, pepper = self.random_coordinates(n)
        image = np.array(sample['image'])
        image[salt[:, 0], salt[:, 1]] = np.array([255, 255, 255])
        image[pepper[:, 0], pepper[:, 1]] = np.array([0, 0, 0])
        sample['image'] = Image.fromarray(image)
        return sample


class RandomRotate(object):
    def __init__(self, max_angle, size=128):
        self.angle = max_angle
        self.size = size

    def __call__(self, sample):
        cur_angle = random.randint(-self.angle, self.angle)
        sample['image'] = TF.rotate(sample['image'], angle=cur_angle)

        shift = float(self.size // 2)
        theta = np.radians(cur_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = torch.tensor(((c, -s), (s, c))).float()
        landmarks = sample['landmarks'].reshape(-1, 2)
        landmarks = torch.matmul(landmarks - shift, R)
        landmarks = landmarks + shift
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

        if split != "test":
            ignore_path = os.path.join(root, 'shift_file.txt')
            with open(ignore_path, "rb") as fp:
                self.ignore_file = pickle.load(fp)

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp), total=num_lines + 1):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reache "images")d start of val part of data
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                if elements[0] in self.ignore_file:
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