import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.cuda.amp  # 添加混合精度训练支持
from utils.utils import print_and_save, shuffling, epoch_time
# from network.dual_model import ConDSeg, TeacherConDSeg
from network.dual_model import ICFSeg, TeacherICFDSeg
from utils.metrics import DiceBCELoss
import itertools
from torch.utils.data.sampler import Sampler
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_engine_ISIC2017 import load_data, evaluate, DATASET, calculate_metrics
from utils.run_engine_stage import BinaryConsistencyLoss
from tqdm import tqdm


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 初始化SAM-Med2D模型
def initialize_sammed2d():
    device = "cuda"

    # 创建必要的参数对象
    class ModelArgs:
        def __init__(self):
            self.image_size = 256
            self.vit_patch_size = 16
            self.encoder_depth = 12
            self.encoder_embed_dim = 768
            self.encoder_num_heads = 12
            self.encoder_global_attn_indexes = [2, 5, 8, 11]
            self.sam_checkpoint = "weights/sammed2d/sam-med2d_b.pth"  # 修改为sam_checkpoint
            self.device = device
            # SAM-Med2D特定参数
            self.encoder_adapter = True
            self.prompt_encoder_adapter = True
            self.mask_decoder_adapter = True
            self.use_adapter = True
            self.adapter_type = "normal"
            self.use_prompt_encoder = True

    args = ModelArgs()

    try:
        # 使用正确的参数创建模型
        print("正在初始化SAM-Med2D模型...")
        model = sam_model_registry["vit_b"](args)

        # 加载权重
        print(f"正在加载权重从: {args.sam_checkpoint}")
        if os.path.exists(args.sam_checkpoint):
            state_dict = torch.load(args.sam_checkpoint)
            if "model" in state_dict:
                model.load_state_dict(state_dict["model"])
            else:
                model.load_state_dict(state_dict)
        else:
            print(f"警告：找不到权重文件 {args.sam_checkpoint}")

        model.to(device=device)
        print("SAM-Med2D模型初始化完成")

        predictor = SammedPredictor(model)
        return predictor

    except Exception as e:
        print(f"初始化SAM-Med2D失败: {str(e)}")
        raise


# 添加生成点的函数
def generate_points_from_prediction_mrs(pred_mask, num_points=20, num_samples=10):
    """
    MRS: Multiple Random Sampling strategy
    Args:
        pred_mask: 预测掩码 [C, H, W] 或 [H, W]
        num_points: 每次采样的点数 M
        num_samples: 采样次数 K
    Returns:
        points_list: K组点坐标
        labels_list: K组标签
    """
    # 确保pred_mask是2D的
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask.squeeze(0)  # 移除channel维度

    h, w = pred_mask.shape

    # 将预测掩码转换为二值图
    binary_mask = (pred_mask > 0.5).float()

    # 获取前景像素坐标
    coords = torch.nonzero(binary_mask)

    points_list = []
    labels_list = []

    if len(coords) > 0:
        for _ in range(num_samples):  # K次采样
            # 随机选择M个前景点
            if len(coords) >= num_points:
                indices = torch.randperm(len(coords))[:num_points]
                selected_coords = coords[indices]
            else:
                # 如果前景点不够，重复使用现有点
                indices = torch.randint(0, len(coords), (num_points,))
                selected_coords = coords[indices]

            # 归一化坐标
            points = torch.zeros((num_points, 2)).cuda()
            points[:, 0] = selected_coords[:, 1].float() / w  # x坐标
            points[:, 1] = selected_coords[:, 0].float() / h  # y坐标

            points_list.append(points)
            labels_list.append(torch.ones(num_points).cuda())
    else:
        # 如果没有前景像素，使用网格采样
        for _ in range(num_samples):
            points = torch.zeros((num_points, 2)).cuda()
            for i in range(num_points):
                points[i, 0] = torch.rand(1).cuda()  # 随机x坐标
                points[i, 1] = torch.rand(1).cuda()  # 随机y坐标
            points_list.append(points)
            labels_list.append(torch.ones(num_points).cuda())

    return points_list, labels_list


# 添加处理SAM-Med2D预测的函数
def process_sammed2d_prediction_mrs(unlabeled_volume_batch, output1_soft, output2_soft, sammed2d_predictor, labeled_bs,
                                    M=20, K=10):
    """
    使用MRS策略处理SAM-Med2D预测，同时使用点提示和掩码提示
    """
    sammed2d_masks = []

    # 计算两个分支的平均预测，并分离计算图
    with torch.no_grad():
        avg_pred = (output1_soft[labeled_bs:] + output2_soft[labeled_bs:]) * 0.5

    for i in range(unlabeled_volume_batch.shape[0]):
        # 设置图像
        image = unlabeled_volume_batch[i].cpu()

        # 获取原始图像尺寸
        original_size = image.shape[-2:]  # [H, W]

        # SAM-Med2D的默认输入尺寸是256x256
        image_256 = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        image_256 = (image_256.squeeze(0) * 255).byte().numpy().transpose(1, 2, 0)
        sammed2d_predictor.set_image(image_256)

        # 准备两个掩码提示，使用detach()分离计算图
        mask1 = output1_soft[labeled_bs + i].detach()
        mask2 = output2_soft[labeled_bs + i].detach()

        # 确保掩码维度正确
        if len(mask1.shape) == 3:
            mask1 = mask1.squeeze(0)
        if len(mask2.shape) == 3:
            mask2 = mask2.squeeze(0)

        # 将两个掩码合并为一个（取平均）
        combined_mask = ((mask1 + mask2) / 2 > 0.5).float()

        # 调整掩码大小为64x64 (SAM-Med2D的image_embedding_size是256/4=64)
        combined_mask_64 = F.interpolate(combined_mask.unsqueeze(0).unsqueeze(0),
                                         size=(64, 64),
                                         mode='nearest')

        # 转换为numpy数组，保持4D格式 [1, 1, H, W]
        mask_input = combined_mask_64.cpu().numpy()

        # 获取当前预测的采样点
        current_pred = avg_pred[i]
        if len(current_pred.shape) == 3:
            current_pred = current_pred.squeeze(0)

        # 生成点提示，并将坐标缩放到256x256
        points_list, labels_list = generate_points_from_prediction_mrs(current_pred, M, K)

        # 存储K次预测结果
        pred_masks = []

        # K次推理 - 使用点提示和掩码提示
        for points, labels in zip(points_list, labels_list):
            try:
                # 将点坐标缩放到256x256
                points_256 = points.clone()
                points_256[:, 0] *= 256 / original_size[1]  # x坐标缩放
                points_256[:, 1] *= 256 / original_size[0]  # y坐标缩放

                # 确保点坐标在[0, 255]范围内
                points_256 = torch.clamp(points_256, 0, 255)

                # 同时使用点提示和掩码提示进行预测
                masks, _, _ = sammed2d_predictor.predict(
                    point_coords=points_256.detach().cpu().numpy(),
                    point_labels=labels.detach().cpu().numpy(),
                    mask_input=mask_input,  # 使用[1, 1, 64, 64]格式的掩码
                    multimask_output=False
                )

                # 将预测结果缩放回原始尺寸
                mask_tensor = torch.from_numpy(masks[0]).float().unsqueeze(0).unsqueeze(0)
                mask_original = F.interpolate(mask_tensor,
                                           size=original_size,
                                           mode='bilinear',
                                           align_corners=False).squeeze(0).squeeze(0)
                pred_masks.append(mask_original.cuda())
            except Exception as e:
                print(f"Error during prediction: {e}")
                print(f"Points shape: {points.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Mask input shape: {mask_input.shape}")
                print(f"Points_256 range: ({points_256.min()}, {points_256.max()})")
                print(f"Mask input range: ({mask_input.min()}, {mask_input.max()})")
                print(f"Mask input type: {type(mask_input)}")
                raise e

        # 计算K次预测的平均值
        avg_mask = torch.stack(pred_masks).mean(0)
        sammed2d_masks.append(avg_mask)

    return torch.stack(sammed2d_masks, dim=0)


# 添加SAM-Med2D一致性损失
def sammed2d_consistency_loss(student_pred1, student_pred2, sammed2d_mask):
    student_avg = (student_pred1 + student_pred2) * 0.5

    # 确保sammed2d_mask有正确的维度 [B, 1, H, W]
    if len(sammed2d_mask.shape) == 3:
        sammed2d_mask = sammed2d_mask.unsqueeze(1)

    # 确保student_avg有正确的维度 [B, 1, H, W]
    if len(student_avg.shape) == 3:
        student_avg = student_avg.unsqueeze(1)

    dice_loss = dice1_loss(student_avg, sammed2d_mask)
    mse_loss = F.mse_loss(student_avg, sammed2d_mask)

    # 固定权重
    dice_weight = 0.65
    mse_weight = 0.35

    consistency_loss = dice_weight * dice_loss + mse_weight * mse_loss
    return consistency_loss


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class SemiSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.is_labeled = [True] * len(labeled_dataset) + [False] * len(unlabeled_dataset)

    def __getitem__(self, index):
        if index < len(self.labeled_dataset):
            return self.labeled_dataset[index], True
        else:
            # 对于无标签数据，返回图像和空标签（实际不会使用）
            image, _ = self.unlabeled_dataset[index - len(self.labeled_dataset)]
            return (image, _), False

    def __len__(self):
        return len(self.labeled_dataset) + len(self.unlabeled_dataset)


def create_semi_supervised_split(train_x, train_y, labeled_ratio=0.1):
    """
    创建半监督学习的数据集划分
    """
    # 确保总是相同的随机划分
    np.random.seed(42)

    num_samples = len(train_x)
    num_labeled = int(num_samples * labeled_ratio)

    # 随机选择有标注的样本索引
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    labeled_indices = indices[:num_labeled]
    unlabeled_indices = indices[num_labeled:]

    # 恢复随机种子
    np.random.seed(int(time.time()))

    return labeled_indices, unlabeled_indices


def update_ema_variables(teacher_model, student_model, alpha=0.99, global_step=None, max_iterations=None):
    """
    用来更新Mean Teacher模型的指数移动平均
    支持教师模型和学生模型有不同结构的情况，以及数据类型不同的情况
    如果提供了global_step和max_iterations参数，则会使用更激进的EMA更新策略
    """
    # 如果提供了global_step，则使用动态alpha值
    if global_step is not None and max_iterations is not None:
        # 更激进的EMA更新
        alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step < max_iterations * 0.3:
            alpha = min(alpha, 0.95)

    # 获取两个模型的状态字典
    teacher_dict = teacher_model.state_dict()
    student_dict = student_model.state_dict()

    # 只更新共享的参数
    for name in teacher_dict:
        if name in student_dict and teacher_dict[name].shape == student_dict[name].shape:
            # 检查数据类型 - 如果Long和Float不匹配，进行适当处理
            if teacher_dict[name].dtype != student_dict[name].dtype:
                if teacher_dict[name].dtype == torch.long or teacher_dict[name].dtype == torch.int64:
                    # 对于整数类型的参数，不应该使用EMA更新，而应该直接复制
                    if student_dict[name].dtype == torch.float or student_dict[name].dtype == torch.float32:
                        continue  # 跳过这个参数，避免类型不匹配错误
                else:
                    # 对于Float类型的参数，使用正常的EMA更新，但先转换类型
                    try:
                        teacher_dict[name].mul_(alpha).add_(student_dict[name].to(teacher_dict[name].dtype),
                                                            alpha=1 - alpha)
                    except:
                        continue  # 如果转换失败，跳过这个参数
            else:
                # 类型相同，直接更新
                try:
                    teacher_dict[name].mul_(alpha).add_(student_dict[name], alpha=1 - alpha)
                except:
                    continue  # 如果更新失败，跳过这个参数

    # 加载更新后的状态字典
    teacher_model.load_state_dict(teacher_dict)


class ConsistencyLoss(nn.Module):
    """
    使用BinaryConsistencyLoss计算一致性损失
    """

    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        self.binary_consistency = BinaryConsistencyLoss()

    def forward(self, pred1, pred2):
        return self.binary_consistency(pred1, pred2)


def dice1_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def train_semi(student_model, teacher_model, loader, optimizer, loss_fn, device, sammed2d_predictor=None,
               ema_decay=0.99, consistency_weight=10.0, labeled_bs=2, sammed2d_weight=0.2, use_sammed2d=False):
    student_model.train()
    teacher_model.eval()  # 教师模型始终处于评估模式

    epoch_loss = 0.0
    epoch_supervised_loss = 0.0
    epoch_consistency_loss = 0.0
    epoch_sammed2d_loss = 0.0

    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    num_labeled_samples = 0
    num_unlabeled_samples = 0

    # 创建一致性损失函数
    consistency_loss_fn = ConsistencyLoss()

    # 图像增强函数
    strong_augmentation = A.Compose([
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),  # 降低模糊强度
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),  # 降低噪声强度
    ])

    for i, data in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        # 定期清理GPU缓存
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        (x, (y1, y2)) = data

        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        y2 = y2.to(device, dtype=torch.float32)

        batch_size = x.size(0)

        # 前labeled_bs个样本是有标签的，其余的是无标签的
        optimizer.zero_grad()

        # 1. 处理有标签数据
        supervised_loss = torch.tensor(0.0, device=device)
        if labeled_bs > 0:
            # 获取有标签数据 (前labeled_bs个)
            labeled_x = x[:labeled_bs]
            labeled_y1 = y1[:labeled_bs]
            labeled_y2 = y2[:labeled_bs]

            # 学生模型对有标签数据前向传播 - 注意现在有两个mask预测
            labeled_mask_pred, labeled_mask_pred_trans, labeled_fg_pred, labeled_bg_pred, labeled_uc_pred = student_model(
                labeled_x)

            # 对两个mask预测结果都计算监督损失
            loss_mask = loss_fn(labeled_mask_pred, labeled_y1)
            loss_mask_trans = loss_fn(labeled_mask_pred_trans, labeled_y1)

            # 其他损失保持不变
            loss_fg = loss_fn(labeled_fg_pred, labeled_y1)
            loss_bg = loss_fn(labeled_bg_pred, labeled_y2)

            # 按照run_engine.py中的方式计算beta系数
            beta1 = 1 / (torch.tanh(
                labeled_fg_pred.sum() / (labeled_fg_pred.shape[2] * labeled_fg_pred.shape[3])) + 1e-15)
            beta2 = 1 / (torch.tanh(
                labeled_bg_pred.sum() / (labeled_bg_pred.shape[2] * labeled_bg_pred.shape[3])) + 1e-15)
            beta1 = beta1.to(device)
            beta2 = beta2.to(device)

            preds = torch.stack([labeled_fg_pred, labeled_bg_pred, labeled_uc_pred], dim=1)
            probs = F.softmax(preds, dim=1)
            prob_fg, prob_bg, prob_uc = probs[:, 0], probs[:, 1], probs[:, 2]

            # 使用run_engine.py中定义的complementary_loss函数
            loss_comp = (prob_fg * prob_bg).sum() + (prob_fg * prob_uc).sum() + (prob_bg * prob_uc).sum()
            num_pixels = prob_fg.size(0) * prob_fg.size(2) * prob_fg.size(3)  # B * H * W
            loss_comp = loss_comp / num_pixels
            loss_comp = loss_comp.to(device)

            # 总监督损失 - 加入新的mask预测结果的损失
            supervised_loss = loss_mask + loss_mask_trans + beta1 * loss_fg + beta2 * loss_bg + loss_comp
            num_labeled_samples += labeled_bs

            # 在有标签样本上计算指标 - 使用原始mask_pred计算指标
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for idx in range(labeled_mask_pred.size(0)):
                yt, yp = labeled_y1[idx], labeled_mask_pred[idx]
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            if batch_jac:  # 仅当有标注样本时更新指标
                epoch_jac += np.mean(batch_jac) * len(batch_jac)
                epoch_f1 += np.mean(batch_f1) * len(batch_jac)
                epoch_recall += np.mean(batch_recall) * len(batch_jac)
                epoch_precision += np.mean(batch_precision) * len(batch_jac)

        # 2. 处理无标签数据
        consistency_loss_value = torch.tensor(0.0, device=device)
        sammed2d_loss_value = torch.tensor(0.0, device=device)

        if batch_size > labeled_bs:
            # 获取无标签数据 (从labeled_bs之后)
            unlabeled_x = x[labeled_bs:]

            # 为无标签数据生成弱增强和强增强版本
            unlabeled_x_weak = unlabeled_x  # 原始版本作为弱增强

            # 生成强增强版本
            unlabeled_x_strong = unlabeled_x.clone().cpu().numpy().transpose(0, 2, 3, 1)
            for j in range(unlabeled_x.size(0)):
                unlabeled_x_strong[j] = strong_augmentation(image=unlabeled_x_strong[j])["image"]
            unlabeled_x_strong = torch.tensor(unlabeled_x_strong.transpose(0, 3, 1, 2)).to(device, dtype=torch.float32)

            # 学生模型对弱增强无标签数据前向传播 - 注意现在有两个mask预测
            unlabeled_mask_pred, unlabeled_mask_pred_trans, unlabeled_fg_pred, unlabeled_bg_pred, _ = student_model(
                unlabeled_x_weak)

            # 将两个预测结果取平均
            unlabeled_mask_pred_avg = (unlabeled_mask_pred + unlabeled_mask_pred_trans) / 2.0

            # 教师模型对强增强的无标签数据进行前向传播 - 教师模型还是原来的模型，只有一个mask预测
            with torch.no_grad():
                t_mask_pred, t_fg_pred, t_bg_pred, _ = teacher_model(unlabeled_x_strong)

            
            confidence_threshold = 0.8  
            confidence_mask = (t_mask_pred > confidence_threshold) | (t_mask_pred < (1 - confidence_threshold))
            
            confidence_weight = torch.ones_like(t_mask_pred)
            confidence_weight = torch.where(t_mask_pred > confidence_threshold, t_mask_pred, confidence_weight)
            confidence_weight = torch.where(t_mask_pred < (1 - confidence_threshold), 1 - t_mask_pred,
                                            confidence_weight)

            cons_loss_mask = F.mse_loss(unlabeled_mask_pred_avg * confidence_mask, t_mask_pred * confidence_mask,
                                        reduction='mean')
            cons_loss_fg = F.mse_loss(unlabeled_fg_pred * confidence_mask, t_fg_pred * confidence_mask,
                                      reduction='mean')
            cons_loss_bg = F.mse_loss(unlabeled_bg_pred * confidence_mask, t_bg_pred * confidence_mask,
                                      reduction='mean')

      
            consistency_loss_value = cons_loss_mask + cons_loss_fg + cons_loss_bg
            num_unlabeled_samples += (batch_size - labeled_bs)

            if use_sammed2d and sammed2d_predictor is not None:
           
                output1_soft = torch.sigmoid(unlabeled_mask_pred)
                output2_soft = torch.sigmoid(unlabeled_mask_pred_trans)

            
                full_output1_soft = torch.sigmoid(torch.cat([labeled_mask_pred, unlabeled_mask_pred],
                                                            dim=0) if labeled_bs > 0 else unlabeled_mask_pred)
                full_output2_soft = torch.sigmoid(torch.cat([labeled_mask_pred_trans, unlabeled_mask_pred_trans],
                                                            dim=0) if labeled_bs > 0 else unlabeled_mask_pred_trans)

                
                sam_masks = process_sammed2d_prediction_mrs(
                    unlabeled_x,
                    full_output1_soft,
                    full_output2_soft,
                    sammed2d_predictor,
                    labeled_bs,
                    M=20,  
                    K=15  
                )

                sammed2d_loss_value = sammed2d_consistency_loss(
                    output1_soft,
                    output2_soft,
                    sam_masks
                )

                epoch_sammed2d_loss += sammed2d_loss_value.item() * (batch_size - labeled_bs)


        if not isinstance(consistency_loss_value, torch.Tensor):
            consistency_loss_value = torch.tensor(consistency_loss_value, device=device)
        if not isinstance(sammed2d_loss_value, torch.Tensor):
            sammed2d_loss_value = torch.tensor(sammed2d_loss_value, device=device)

        loss = supervised_loss
        if batch_size > labeled_bs:
            loss = loss + consistency_weight * consistency_loss_value
            if use_sammed2d and sammed2d_predictor is not None:
                loss = loss + sammed2d_weight * sammed2d_loss_value

        loss.backward()

        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

        optimizer.step()

        update_ema_variables(teacher_model, student_model, alpha=ema_decay, global_step=i, max_iterations=len(loader))

        epoch_loss += loss.item() * batch_size
        epoch_supervised_loss += supervised_loss.item() * batch_size
        epoch_consistency_loss += consistency_loss_value.item() * batch_size

        del x, y1, y2
        if labeled_bs > 0:
            del labeled_x, labeled_y1, labeled_y2, labeled_mask_pred, labeled_mask_pred_trans, labeled_fg_pred, labeled_bg_pred, labeled_uc_pred
        if batch_size > labeled_bs:
            del unlabeled_x, unlabeled_x_weak, unlabeled_x_strong, unlabeled_mask_pred, unlabeled_mask_pred_trans, unlabeled_fg_pred, unlabeled_bg_pred
            del t_mask_pred, t_fg_pred, t_bg_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    epoch_loss = epoch_loss / len(loader.dataset)
    epoch_supervised_loss = epoch_supervised_loss / len(loader.dataset)
    epoch_consistency_loss = epoch_consistency_loss / len(loader.dataset)

    if num_unlabeled_samples > 0:
        epoch_sammed2d_loss = epoch_sammed2d_loss / num_unlabeled_samples
    else:
        epoch_sammed2d_loss = 0.0

    if num_labeled_samples > 0:
        epoch_jac = epoch_jac / num_labeled_samples
        epoch_f1 = epoch_f1 / num_labeled_samples
        epoch_recall = epoch_recall / num_labeled_samples
        epoch_precision = epoch_precision / num_labeled_samples

    return epoch_loss, epoch_supervised_loss, epoch_consistency_loss, epoch_sammed2d_loss, [epoch_jac, epoch_f1,
                                                                                            epoch_recall,
                                                                                            epoch_precision]



def evaluate_student(model, loader, loss_fn, device):
    """
    评估包含两个预测结果的学生模型
    """
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for data in tqdm(loader, desc="Validation", total=len(loader)):
            x, (y1, y2) = data

            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)

          
            mask_pred, mask_pred_trans, fg_pred, bg_pred, _ = model(x)

            mask_pred_avg = (mask_pred + mask_pred_trans) / 2.0

            loss = loss_fn(mask_pred_avg, y1)

            epoch_loss += loss.item() * x.size(0)

            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for i in range(mask_pred_avg.size(0)):
                yt, yp = y1[i], mask_pred_avg[i]
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac) * len(batch_jac)
            epoch_f1 += np.mean(batch_f1) * len(batch_f1)
            epoch_recall += np.mean(batch_recall) * len(batch_recall)
            epoch_precision += np.mean(batch_precision) * len(batch_precision)


    epoch_loss = epoch_loss / len(loader.dataset)
    epoch_jac = epoch_jac / len(loader.dataset)
    epoch_f1 = epoch_f1 / len(loader.dataset)
    epoch_recall = epoch_recall / len(loader.dataset)
    epoch_precision = epoch_precision / len(loader.dataset)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":

    gpu_id = 1  
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    dataset_name = 'ISIC-2018'  
    val_name = None
    labeled_ratio = 0.1  

    seed = 42
    my_seeding(seed)

    image_size = 256
    size = (image_size, image_size)
    batch_size = 4
    labeled_bs = 2  
    num_epochs = 300
    lr = 1e-4
    lr_backbone = 1e-5
    early_stopping_patience = 100
    ema_decay = 0.99  
    consistency_weight = 0.1  
    rampup_length = 200  

    use_sammed2d = True 
    sammed2d_weight = 0.2 
    sammed2d_epoch_start = 0  

    pretrained_backbone = None
    resume_path = None

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"semi_{dataset_name}_labeled{int(labeled_ratio * 100)}pct_{current_time}"

    base_dir = "data"
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join("run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    teacher_checkpoint_path = os.path.join(save_dir, "teacher_checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLabeled Batch Size: {labeled_bs}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Labeled Ratio: {labeled_ratio}\n"
    hyperparameters_str += f"EMA Decay: {ema_decay}\n"
    hyperparameters_str += f"Consistency Weight: {consistency_weight}\n"
    hyperparameters_str += f"Consistency Rampup Length: {rampup_length}\n"
    hyperparameters_str += f"Use SAM-Med2D: {use_sammed2d}\n"
    hyperparameters_str += f"SAM-Med2D Initial Weight: {sammed2d_weight}\n"
    hyperparameters_str += f"SAM-Med2D Weight Strategy: Ramp-down with λ_s=0.1⋅e^(−5(t/t_max)) from start\n"
    hyperparameters_str += f"SAM-Med2D Start Epoch: {sammed2d_epoch_start} (from beginning)\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ 数据增强: Transforms """
    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ], additional_targets={'background': 'mask'})

    """ 数据集 """
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)

    labeled_indices, unlabeled_indices = create_semi_supervised_split(train_x, train_y, labeled_ratio)

    labeled_x = [train_x[i] for i in labeled_indices]
    labeled_y = [train_y[i] for i in labeled_indices]
    unlabeled_x = [train_x[i] for i in unlabeled_indices]
    unlabeled_y = [train_y[i] for i in unlabeled_indices] 

    data_str = f"Dataset Size:\nTotal Train: {len(train_x)}\nLabeled: {len(labeled_x)} ({labeled_ratio * 100:.1f}%)\nUnlabeled: {len(unlabeled_x)}\nValid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ 数据集和加载器 """
    labeled_dataset = DATASET(labeled_x, labeled_y, (image_size, image_size), transform=transform)
    unlabeled_dataset = DATASET(unlabeled_x, unlabeled_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    full_dataset = torch.utils.data.ConcatDataset([labeled_dataset, unlabeled_dataset])

    # 创建双流批采样器，确保每个批次包含固定数量的有标签和无标签数据
    batch_sampler = TwoStreamBatchSampler(
        primary_indices=list(range(len(labeled_dataset))),
        secondary_indices=list(range(len(labeled_dataset), len(full_dataset))),
        batch_size=batch_size,
        secondary_batch_size=batch_size - labeled_bs
    )

    # 数据加载器 - 使用批采样器
    train_loader = DataLoader(
        dataset=full_dataset,
        batch_sampler=batch_sampler,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ 模型 """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 学生模型 (主模型)
    student_model = ICFSeg()

    # 教师模型 (EMA模型) - 使用原始版本的网络结构
    teacher_model = TeacherICFDSeg()

    # 初始化SAM-Med2D模型
    sammed2d_predictor = None
    if use_sammed2d:
        try:
            sammed2d_predictor = initialize_sammed2d()
            print("SAM-Med2D模型初始化成功！")
        except Exception as e:
            print(f"初始化SAM-Med2D失败，将不使用SAM-Med2D: {e}")
            use_sammed2d = False

    # 初始化教师模型参数
    # 由于两个模型结构不同，我们需要分别处理共享参数
    student_dict = student_model.state_dict()
    teacher_dict = teacher_model.state_dict()

    # 复制共享部分的参数
    for name in teacher_dict:
        if name in student_dict:
            teacher_dict[name].copy_(student_dict[name])

    # 确保教师模型参数不需要梯度
    for param in teacher_model.parameters():
        param.requires_grad = False

    if pretrained_backbone:
        saved_weights = torch.load(pretrained_backbone)
        for name, param in student_model.named_parameters():
            if name.startswith('layer0') or name.startswith('layer1') or name.startswith('layer2') or name.startswith(
                    'layer3'):
                param.data = saved_weights[name]
        # 同步教师模型
        for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
            param_t.data.copy_(param_s.data)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        student_model.load_state_dict(checkpoint)
        # 同步教师模型
        for param_t, param_s in zip(teacher_model.parameters(), student_model.parameters()):
            param_t.data.copy_(param_s.data)

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)

    # 参数分组
    param_groups = [
        {'params': [], 'lr': lr_backbone},
        {'params': [], 'lr': lr}
    ]

    for name, param in student_model.named_parameters():
        if name.startswith('layer0') or name.startswith('layer1') or name.startswith('layer2') or name.startswith(
                'layer3'):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)

    assert len(param_groups[0]['params']) > 0, "Layer group is empty!"
    assert len(param_groups[1]['params']) > 0, "Rest group is empty!"

    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ 训练模型 """
    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write(
            "epoch,train_loss,supervised_loss,consistency_loss,sammed2d_loss,train_mIoU,train_f1,train_recall,train_precision,valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n")

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # 计算一致性权重的ramp-up
        current_consistency_weight = consistency_weight
        if epoch < rampup_length:
            current_consistency_weight = min(0.1, consistency_weight * (epoch / rampup_length))
        else:
            current_consistency_weight = 0.1

        # 决定是否在当前epoch使用SAM-Med2D - 由于从开始就使用，这里可以简化
        current_use_sammed2d = use_sammed2d

        # 计算SAM-Med2D权重的ramp-down（指数衰减）
        current_sammed2d_weight = 0.0
        if current_use_sammed2d:
            # 根据公式λ_s=0.1⋅e^(−5(t/t_max))计算权重
            # 其中t是当前epoch，t_max是最大epoch数
            t = epoch
            t_max = num_epochs
            current_sammed2d_weight = sammed2d_weight * np.exp(-5 * (t / t_max))
            # 当权重下降到0.040时固定不变
            # if current_sammed2d_weight < 0.0461:
            #     current_sammed2d_weight = 0.0461

            print(f"Epoch {epoch + 1}: 使用SAM-Med2D辅助训练，当前权重: {current_sammed2d_weight:.4f} (ramp-down)")

        # 训练 - 传入当前的SAM-Med2D权重
        train_loss, supervised_loss, consistency_loss, sammed2d_loss, train_metrics = train_semi(
            student_model, teacher_model, train_loader, optimizer, loss_fn, device,
            sammed2d_predictor=sammed2d_predictor if current_use_sammed2d else None,
            ema_decay=ema_decay,
            consistency_weight=current_consistency_weight,
            labeled_bs=labeled_bs,
            sammed2d_weight=current_sammed2d_weight,  # 使用当前计算的权重
            use_sammed2d=current_use_sammed2d
        )

        # 验证 (使用学生模型)
        valid_loss, valid_metrics = evaluate_student(student_model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoints"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(student_model.state_dict(), checkpoint_path)
            torch.save(teacher_model.state_dict(), teacher_checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Supervised Loss: {supervised_loss:.4f} - Consistency Loss: {consistency_loss:.4f} - SAM-Med2D Loss: {sammed2d_loss:.4f}\n"
        data_str += f"\tTrain mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\tValid Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        data_str += f"\tCurrent Consistency Weight: {current_consistency_weight:.4f} - Using SAM-Med2D: {current_use_sammed2d}"
        # 添加SAM-Med2D权重显示
        if current_use_sammed2d:
            data_str += f" - SAM-Med2D Weight: {current_sammed2d_weight:.4f}"
        data_str += "\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{supervised_loss},{consistency_loss},{sammed2d_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving for last {early_stopping_patience} epochs.\n"
            print_and_save(train_log_path, data_str)
            break

