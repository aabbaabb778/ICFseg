import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .utils import print_and_save, shuffling, epoch_time, calculate_metrics
from tqdm import tqdm

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


def load_names(path, fold):
    images_path = []
    masks_path = []

    # 可能的图像和掩码扩展名
    image_extensions = ['.jpg', '.png', '.jpeg']
    mask_extensions = ['.png', '.jpg', '.jpeg']

    base_image_dir = os.path.join(path, "train", "images")
    base_mask_dir = os.path.join(path, "train", "masks")

    # 确保目录存在
    if not os.path.exists(base_image_dir):
        print(f"Warning: Image directory not found at {base_image_dir}")
    if not os.path.exists(base_mask_dir):
        print(f"Warning: Mask directory not found at {base_mask_dir}")

    # 获取所有图像文件的基本名称（不含扩展名）
    image_files = []
    if os.path.exists(base_image_dir):
        for file in os.listdir(base_image_dir):
            base_name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(base_name)

    for base_name in image_files:
        # 查找匹配的图像文件
        image_found = False
        for ext in image_extensions:
            img_path = os.path.join(base_image_dir, f"{base_name}{ext}")
            if os.path.exists(img_path):
                images_path.append(img_path)
                image_found = True
                break

        if not image_found:
            print(f"Warning: Could not find image file for {base_name}")
            continue

        # 查找匹配的掩码文件，尝试两种命名格式：
        # 1. 与图像同名
        # 2. 图像名称+_segmentation后缀
        mask_found = False

        # 尝试与图像同名的掩码
        for ext in mask_extensions:
            mask_path = os.path.join(base_mask_dir, f"{base_name}{ext}")
            if os.path.exists(mask_path):
                masks_path.append(mask_path)
                mask_found = True
                break

        # 如果没找到，尝试带_segmentation后缀的掩码
        if not mask_found:
            for ext in mask_extensions:
                mask_path = os.path.join(base_mask_dir, f"{base_name}_segmentation{ext}")
                if os.path.exists(mask_path):
                    masks_path.append(mask_path)
                    mask_found = True
                    break

        if not mask_found:
            print(f"Warning: Could not find mask file for {base_name}")
            # 如果找不到掩码但找到了图像，我们移除对应的图像路径
            if image_found:
                images_path.pop()

    # 验证图像和掩码数量一致
    assert len(images_path) == len(
        masks_path), f"Number of images ({len(images_path)}) and masks ({len(masks_path)}) do not match"

    if len(images_path) == 0:
        print("Warning: No valid image-mask pairs found!")
    else:
        print(f"Found {len(images_path)} valid image-mask pairs")

    return images_path, masks_path


def load_val_names(path, fold):
    images_path = []
    masks_path = []

    # 可能的图像和掩码扩展名
    image_extensions = ['.jpg', '.png', '.jpeg']
    mask_extensions = ['.png', '.jpg', '.jpeg']

    base_image_dir = os.path.join(path, "val", "images")
    base_mask_dir = os.path.join(path, "val", "masks")

    # 确保目录存在
    if not os.path.exists(base_image_dir):
        print(f"Warning: Validation image directory not found at {base_image_dir}")
    if not os.path.exists(base_mask_dir):
        print(f"Warning: Validation mask directory not found at {base_mask_dir}")

    # 获取所有图像文件的基本名称（不含扩展名）
    image_files = []
    if os.path.exists(base_image_dir):
        for file in os.listdir(base_image_dir):
            base_name, ext = os.path.splitext(file)
            if ext.lower() in image_extensions:
                image_files.append(base_name)

    for base_name in image_files:
        # 查找匹配的图像文件
        image_found = False
        for ext in image_extensions:
            img_path = os.path.join(base_image_dir, f"{base_name}{ext}")
            if os.path.exists(img_path):
                images_path.append(img_path)
                image_found = True
                break

        if not image_found:
            print(f"Warning: Could not find validation image file for {base_name}")
            continue

        # 查找匹配的掩码文件，尝试两种命名格式：
        # 1. 与图像同名
        # 2. 图像名称+_segmentation后缀
        mask_found = False

        # 尝试与图像同名的掩码
        for ext in mask_extensions:
            mask_path = os.path.join(base_mask_dir, f"{base_name}{ext}")
            if os.path.exists(mask_path):
                masks_path.append(mask_path)
                mask_found = True
                break

        # 如果没找到，尝试带_segmentation后缀的掩码
        if not mask_found:
            for ext in mask_extensions:
                mask_path = os.path.join(base_mask_dir, f"{base_name}_segmentation{ext}")
                if os.path.exists(mask_path):
                    masks_path.append(mask_path)
                    mask_found = True
                    break

        if not mask_found:
            print(f"Warning: Could not find validation mask file for {base_name}")
            # 如果找不到掩码但找到了图像，我们移除对应的图像路径
            if image_found:
                images_path.pop()

    # 验证图像和掩码数量一致
    assert len(images_path) == len(
        masks_path), f"Number of validation images ({len(images_path)}) and masks ({len(masks_path)}) do not match"

    if len(images_path) == 0:
        print("Warning: No valid validation image-mask pairs found!")
    else:
        print(f"Found {len(images_path)} valid validation image-mask pairs")

    return images_path, masks_path


def load_data(path, val_name=None):
    print(f"Loading data from {path}")

    train_names_path = f"{path}/train.txt"
    train_x, train_y = load_names(path, train_names_path)

    if val_name is None:
        valid_names_path = f"{path}/val.txt"
    else:
        valid_names_path = f"{path}/val_{val_name}.txt"

    valid_x, valid_y = load_val_names(path, valid_names_path)

    print(f"Loaded {len(train_x)} training samples and {len(valid_x)} validation samples")
    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        # print("n_samples:", self.n_samples)
        # self.convert_edge=convert_edge
        self.size = size

    def __getitem__(self, index):
        """ Reading Image & Mask """
        try:
            # 获取图像和掩码路径
            image_path = self.images_path[index]
            mask_path = self.masks_path[index]

            # 尝试读取图像
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Failed to load image: {image_path}")
                # 创建空白图像作为替代
                image = np.zeros((256, 256, 3), dtype=np.uint8)

            # 尝试读取掩码
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to load mask: {mask_path}")
                # 创建空白掩码作为替代
                mask = np.zeros((256, 256), dtype=np.uint8)

            background = mask.copy()
            background = 255 - background

            """ Applying Data Augmentation """
            if self.transform is not None:
                augmentations = self.transform(image=image, mask=mask, background=background)
                image = augmentations["image"]
                mask = augmentations["mask"]
                background = augmentations["background"]

            """ Image """
            image = cv2.resize(image, self.size)
            image = np.transpose(image, (2, 0, 1))
            image = image / 255.0

            """ Mask """
            mask = cv2.resize(mask, self.size)
            mask = np.expand_dims(mask, axis=0)
            mask = mask / 255.0

            """ Background """
            background = cv2.resize(background, self.size)
            background = np.expand_dims(background, axis=0)
            background = background / 255.0

            return image, (mask, background)
        except Exception as e:
            print(f"Error loading data at index {index}: {e}")
            # 返回一个空占位符
            image = np.zeros((3, self.size[0], self.size[1]), dtype=np.float32)
            mask = np.zeros((1, self.size[0], self.size[1]), dtype=np.float32)
            background = np.zeros((1, self.size[0], self.size[1]), dtype=np.float32)
            return image, (mask, background)

    def __len__(self):
        return self.n_samples


class BinaryConsistencyLoss(nn.Module):
    def __init__(self):
        super(BinaryConsistencyLoss, self).__init__()

    def forward(self, mask1, mask2):
        mask1_binary = (mask1 > 0.5).float()
        mask2_binary = (mask2 > 0.5).float()

        loss1 = F.binary_cross_entropy(mask1, mask2_binary, reduction='mean')
        loss2 = F.binary_cross_entropy(mask2, mask1_binary, reduction='mean')

        loss = loss1 + loss2

        return loss


def train(model, loader, optimizer, loss_fn, device, consistency_loss_fn=None):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    # 检查是否使用一致性损失
    if consistency_loss_fn is None:
        consistency_loss_fn = BinaryConsistencyLoss()

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred = model(x)

        # 使用增强版本并计算一致性损失
        x_aug = x.clone()
        # 简单的数据增强，作为示例
        if random.random() > 0.5:
            x_aug = torch.flip(x_aug, [3])  # 水平翻转

        mask_pred_aug = model(x_aug)

        loss_consistency = consistency_loss_fn(mask_pred, mask_pred_aug)
        loss_mask = loss_fn(mask_pred, y1)
        loss_mask_aug = loss_fn(mask_pred_aug, y1)

        loss = loss_mask + loss_mask_aug + 0.1 * loss_consistency

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss / len(loader)
    epoch_jac = epoch_jac / len(loader)
    epoch_f1 = epoch_f1 / len(loader)
    epoch_recall = epoch_recall / len(loader)
    epoch_precision = epoch_precision / len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)

            mask_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)

            loss = loss_mask

            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss / len(loader)
        epoch_jac = epoch_jac / len(loader)
        epoch_f1 = epoch_f1 / len(loader)
        epoch_recall = epoch_recall / len(loader)
        epoch_precision = epoch_precision / len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
