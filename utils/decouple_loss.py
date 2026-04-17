import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupleLoss(nn.Module):
    """
    数据解耦损失模块
    将学生模型的两个预测输出解耦为高置信、分歧、低置信三个区域
    并对每个区域采用不同的优化策略
    """

    def __init__(self, tau=0.8, q=2.0, r=0.5, cps_weight=1.0, mm_weight=1.0, rc_weight=1.0):
        super(DecoupleLoss, self).__init__()
        self.tau = tau  # 置信度阈值
        self.q = q  # 平滑温度参数 (q > 1)
        self.r = r  # 锐化温度参数 (r < 1)

        # 各损失权重
        self.cps_weight = cps_weight  # 交叉伪监督损失权重
        self.mm_weight = mm_weight  # 互匹配损失权重
        self.rc_weight = rc_weight  # 强化一致性损失权重

        # Dice损失函数
        self.dice_loss = DiceLoss()

    def forward(self, PA, PB, consistency_weight=0.1):
        """
        计算解耦损失
        Args:
            PA: 解码器A的预测输出 [B, 1, H, W]
            PB: 解码器B的预测输出 [B, 1, H, W]
            consistency_weight: 当前一致性权重
        Returns:
            total_loss: 总解耦损失
            loss_dict: 包含各项损失的字典
        """
        # 获取概率值
        prob_A = torch.sigmoid(PA)
        prob_B = torch.sigmoid(PB)

        # 创建区域掩码
        high_conf_mask, div_mask, low_conf_mask = self._create_region_masks(prob_A, prob_B)

        # 计算各项损失
        cps_loss = self._cross_pseudo_supervision_loss(PA, PB, prob_A, prob_B, high_conf_mask)
        mm_loss = self._mutual_matching_loss(PA, PB, div_mask)
        rc_loss = self._reinforcement_consistency_loss(PA, PB, low_conf_mask)

        # 按照自定义加权方式
        total_loss = (
                consistency_weight * cps_loss +
                (1 - consistency_weight) * mm_loss +
                2 * rc_loss
        )

        loss_dict = {
            'cps_loss': cps_loss.item(),
            'mm_loss': mm_loss.item(),
            'rc_loss': rc_loss.item(),
            'total_decouple_loss': total_loss.item(),
            'high_conf_ratio': high_conf_mask.float().mean().item(),
            'div_ratio': div_mask.float().mean().item(),
            'low_conf_ratio': low_conf_mask.float().mean().item()
        }

        return total_loss, loss_dict

    def _create_region_masks(self, prob_A, prob_B):
        """
        创建三个区域的掩码

        Returns:
            high_conf_mask: 高置信区域掩码
            div_mask: 分歧区域掩码
            low_conf_mask: 低置信区域掩码
        """
        # 高置信区域：两个分支都大于tau或都小于1-tau
        high_conf_mask = ((prob_A > self.tau) & (prob_B > self.tau)) | \
                         ((prob_A < (1 - self.tau)) & (prob_B < (1 - self.tau)))

        # 分歧区域：一个大于tau，另一个小于1-tau
        div_mask = ((prob_A > self.tau) & (prob_B < (1 - self.tau))) | \
                   ((prob_A < (1 - self.tau)) & (prob_B > self.tau))

        # 低置信区域：两个分支都在[tau, 1-tau]之间
        low_conf_mask = (~high_conf_mask) & (~div_mask)

        return high_conf_mask, div_mask, low_conf_mask

    def _cross_pseudo_supervision_loss(self, PA, PB, prob_A, prob_B, high_conf_mask):
        """
        交叉伪监督损失

        Args:
            PA, PB: 原始预测输出
            prob_A, prob_B: 概率值
            high_conf_mask: 高置信区域掩码

        Returns:
            cps_loss: 交叉伪监督损失
        """
        if high_conf_mask.sum() == 0:
            return torch.tensor(0.0, device=PA.device)

        # 生成伪标签 (argmax)
        pseudo_label_A = (prob_A > 0.5).float().detach()
        pseudo_label_B = (prob_B > 0.5).float().detach()

        # 交叉伪监督：A用B的伪标签，B用A的伪标签
        cps_loss_A = F.binary_cross_entropy_with_logits(
            PA[high_conf_mask],
            pseudo_label_B[high_conf_mask]
        )
        cps_loss_B = F.binary_cross_entropy_with_logits(
            PB[high_conf_mask],
            pseudo_label_A[high_conf_mask]
        )

        return cps_loss_A + cps_loss_B

    def _mutual_matching_loss(self, PA, PB, div_mask):
        """
        互匹配损失 (分歧区域)

        Args:
            PA, PB: 原始预测输出
            div_mask: 分歧区域掩码

        Returns:
            mm_loss: 互匹配损失
        """
        if div_mask.sum() == 0:
            return torch.tensor(0.0, device=PA.device)

        # 平滑处理 (提高熵值)
        prob_A_div = torch.sigmoid(PA / self.q)
        prob_B_div = torch.sigmoid(PB / self.q)

        # 只对分歧区域计算Dice损失
        mm_loss = self.dice_loss(prob_A_div[div_mask], prob_B_div[div_mask])

        return mm_loss

    def _reinforcement_consistency_loss(self, PA, PB, low_conf_mask):
        """
        强化一致性损失 (低置信区域)

        Args:
            PA, PB: 原始预测输出
            low_conf_mask: 低置信区域掩码

        Returns:
            rc_loss: 强化一致性损失
        """
        if low_conf_mask.sum() == 0:
            return torch.tensor(0.0, device=PA.device)

        # 锐化处理 (降低熵值)
        prob_A_low = torch.sigmoid(PA / self.r)
        prob_B_low = torch.sigmoid(PB / self.r)

        # MSE损失
        rc_loss = F.mse_loss(prob_A_low[low_conf_mask], prob_B_low[low_conf_mask])

        return rc_loss


class DiceLoss(nn.Module):
    """
    Dice损失函数
    """

    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算Dice损失

        Args:
            pred: 预测值 [N]
            target: 目标值 [N]

        Returns:
            dice_loss: Dice损失
        """
        # 确保输入是一维的
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice_coeff = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice_coeff


def test_decouple_loss():
    """
    测试解耦损失函数
    """
    # 创建测试数据
    batch_size, channels, height, width = 2, 1, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模拟两个解码器的输出
    PA = torch.randn(batch_size, channels, height, width, device=device)
    PB = torch.randn(batch_size, channels, height, width, device=device)

    # 创建解耦损失实例
    decouple_loss = DecoupleLoss(tau=0.7, q=2.0, r=0.5)

    # 计算损失
    total_loss, loss_dict = decouple_loss(PA, PB)

    print("解耦损失测试结果:")
    print(f"总损失: {total_loss.item():.4f}")
    print(f"交叉伪监督损失: {loss_dict['cps_loss']:.4f}")
    print(f"互匹配损失: {loss_dict['mm_loss']:.4f}")
    print(f"强化一致性损失: {loss_dict['rc_loss']:.4f}")
    print(f"高置信区域比例: {loss_dict['high_conf_ratio']:.4f}")
    print(f"分歧区域比例: {loss_dict['div_ratio']:.4f}")
    print(f"低置信区域比例: {loss_dict['low_conf_ratio']:.4f}")


if __name__ == "__main__":
    test_decouple_loss()