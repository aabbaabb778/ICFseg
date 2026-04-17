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

from .utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics, mask_to_bbox
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn.functional as F


def load_names(path, fold):
    images_path = []
    masks_path = []

    # 读取列表文件
    if fold == "train":
        txt_path = os.path.join(path, "train.txt")
    elif fold == "val":
        txt_path = os.path.join(path, "val.txt")
    else:
        txt_path = os.path.join(path, f"{fold}.txt")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到文件: {txt_path}")

    # 读取文件列表
    with open(txt_path, "r") as f:
        for line in f:
            img_name = line.strip()
            if not img_name:  # 跳过空行
                continue

            # 检查图像名称是否包含扩展名
            if not any(img_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                # 尝试不同的扩展名
                possible_img_files = [
                    os.path.join(path, "train" if fold == "train" else "val", "images", img_name + ".jpg"),
                    os.path.join(path, "train" if fold == "train" else "val", "images", img_name + ".png"),
                    os.path.join(path, "train" if fold == "train" else "val", "images", img_name + ".jpeg")
                ]
                img_path = None
                for possible_img in possible_img_files:
                    if os.path.exists(possible_img):
                        img_path = possible_img
                        img_name = os.path.basename(img_path)  # 更新带扩展名的文件名
                        break
                if img_path is None:
                    print(f"警告: 图片文件不存在: {img_name}，将尝试检查原始路径")
                    img_path = os.path.join(path, "train" if fold == "train" else "val", "images", img_name)
            else:
                # 直接使用带扩展名的文件名
                img_path = os.path.join(path, "train" if fold == "train" else "val", "images", img_name)

            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 图片文件不存在: {img_path}，将跳过该样本")
                continue

            # 尝试多种掩码命名格式
            mask_base = os.path.splitext(img_name)[0]
            mask_path = None

            # 检查常规掩码
            possible_mask_files = [
                os.path.join(path, "train" if fold == "train" else "val", "masks", mask_base + ".png"),
                os.path.join(path, "train" if fold == "train" else "val", "masks", mask_base + ".jpg"),
                os.path.join(path, "train" if fold == "train" else "val", "masks", mask_base + "_segmentation.png"),
                os.path.join(path, "train" if fold == "train" else "val", "masks", mask_base + "_segmentation.jpg")
            ]

            for possible_mask in possible_mask_files:
                if os.path.exists(possible_mask):
                    mask_path = possible_mask
                    break

            if mask_path is None:
                print(f"警告: 找不到图片 {img_name} 对应的掩码文件，将跳过该样本")
                continue

            images_path.append(img_path)
            masks_path.append(mask_path)

    print(f"从 {txt_path} 加载了 {len(images_path)} 个样本")
    return images_path, masks_path


def load_data(path, val_name=None):
    """
    加载训练集和验证集数据

    Args:
        path: 数据集根目录
        val_name: 验证集名称，如果为None则使用默认验证集

    Returns:
        训练集和验证集的图像和掩码路径
    """
    # 加载训练集
    train_x, train_y = load_names(path, "train")

    # 加载验证集
    if val_name is not None:
        valid_x, valid_y = load_names(path, val_name)
    else:
        valid_x, valid_y = load_names(path, "val")

    return (train_x, train_y), (valid_x, valid_y)


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def preprocess_mask(self, mask):
        """特殊处理ISIC-2017数据集的掩码"""
        # ISIC-2017掩码可能是RGB格式，需要转换为灰度
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # 转换为灰度
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 确保掩码是二值的
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def __getitem__(self, index):
        """ Reading Image & Mask """
        try:
            # 读取图像
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            if image is None:
                print(f"警告: 无法读取图像: {self.images_path[index]}")
                # 创建一个空白图像作为替代
                image = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)

            # 读取掩码
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_UNCHANGED)  # 使用UNCHANGED以支持各种格式
            if mask is None:
                print(f"警告: 无法读取掩码: {self.masks_path[index]}")
                # 创建一个空白掩码作为替代
                mask = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)
            else:
                # 处理掩码
                mask = self.preprocess_mask(mask)

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
            print(f"处理数据时出错 (索引 {index}): {str(e)}")
            # 返回空白数据
            image = np.zeros((3, self.size[0], self.size[1]), dtype=np.float32)
            mask = np.zeros((1, self.size[0], self.size[1]), dtype=np.float32)
            background = np.zeros((1, self.size[0], self.size[1]), dtype=np.float32)
            return image, (mask, background)

    def __len__(self):
        return self.n_samples


def complementary_loss(prob_fg, prob_bg, prob_uc):
    loss = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
    num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
    normalized_loss = loss / num_pixels
    return normalized_loss


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        mask_pred, fg_pred, bg_pred, uc_pred = model(x)

        loss_mask = loss_fn(mask_pred, y1)
        loss_fg = loss_fn(fg_pred, y1)
        loss_bg = loss_fn(bg_pred, y2)

        beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
        beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
        beta1 = beta1.to(device)
        beta2 = beta2.to(device)
        preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
        probs = F.softmax(preds, dim=1)
        prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

        loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
        loss_comp = loss_comp.to(device)
        loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp
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

            mask_pred, fg_pred, bg_pred, uc_pred = model(x)

            loss_mask = loss_fn(mask_pred, y1)
            loss_fg = loss_fn(fg_pred, y1)
            loss_bg = loss_fn(bg_pred, y2)

            beta1 = 1 / (torch.tanh(fg_pred.sum() / (fg_pred.shape[2] * fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(bg_pred.sum() / (bg_pred.shape[2] * bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([fg_pred, bg_pred, uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            loss_comp = complementary_loss(prob_fg, prob_bg, prob_uc)
            loss_comp = loss_comp.to(device)

            loss = loss_mask + beta1 * loss_fg + beta2 * loss_bg + loss_comp

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
