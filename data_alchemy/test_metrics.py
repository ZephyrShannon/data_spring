from logging import Logger

import torch

class ValidationMetrics:
    """
    验证集指标计算器（正确方法）
    """

    def __init__(self, num_heads, device):
        self.num_heads = num_heads
        self.reset(device)

    def reset(self, device):
        """重置累积器"""
        self.tp = torch.zeros(self.num_heads, device=device)
        self.fp = torch.zeros(self.num_heads, device=device)
        self.fn = torch.zeros(self.num_heads, device=device)
        self.tn = torch.zeros(self.num_heads, device=device)
        self.total_samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """
        更新累积统计

        Args:
            logits: (B, H) 模型输出的 logits
            targets: (B, H) 真实标签
            threshold: 分类阈值
        """
        # 转换为预测
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        # 更新混淆矩阵
        batch_tp = (preds * targets).sum(dim=0)
        batch_fp = (preds * (1 - targets)).sum(dim=0)
        batch_fn = ((1 - preds) * targets).sum(dim=0)
        batch_tn = ((1 - preds) * (1 - targets)).sum(dim=0)

        self.tp += batch_tp
        self.fp += batch_fp
        self.fn += batch_fn
        self.tn += batch_tn
        self.total_samples += targets.size(0)

    def compute_metrics(self, epsilon: float = 1e-8) -> dict:
        """
        计算最终指标
        """
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        accuracy = (self.tp + self.tn) / (self.total_samples + epsilon)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'total_samples': self.total_samples
        }

    def print_metrics(self, head_names, logger: Logger):
        """打印指标"""
        metrics = self.compute_metrics()

        if head_names is None:
            head_names = [f"Head_{i}" for i in range(self.num_heads)]

        logger.info("验证集评估结果")
        logger.info(f"总样本数: {self.total_samples}")

        logger.info(f"{'Head':<12} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Accuracy':>10}")
        for i in range(self.num_heads):
            logger.info(f"{head_names[i]:<12} | {metrics['precision'][i]:>10.4f} | "
                  f"{metrics['recall'][i]:>10.4f} | {metrics['f1'][i]:>10.4f} | "
                  f"{metrics['accuracy'][i]:>10.4f}")

        logger.info(f"{'平均':<12} | {metrics['precision'].mean():>10.4f} | "
              f"{metrics['recall'].mean():>10.4f} | {metrics['f1'].mean():>10.4f} | "
              f"{metrics['accuracy'].mean():>10.4f}")


# 使用示例
def demonstrate_correct_validation():
    """
    演示正确的验证流程
    """

    print("\n" + "=" * 70)
    print("正确的验证流程演示")
    print("=" * 70)

    # 模拟数据
    torch.manual_seed(42)
    num_heads = 2
    num_batches = 5
    batch_size = 32

    # 创建指标计算器
    metrics_calculator = ValidationMetrics(num_heads)

    # 模拟验证循环
    for batch_idx in range(num_batches):
        # 模拟模型输出和标签
        logits = torch.randn(batch_size, num_heads)
        targets = torch.randint(0, 2, (batch_size, num_heads)).float()

        # 更新累积统计
        metrics_calculator.update(logits, targets)

        # 可选：每个batch打印进度
        print(f"Batch {batch_idx + 1}: 已处理 {metrics_calculator.total_samples} 个样本")

    # 最终打印结果
    head_names = ['long_5m', 'short_5m']
    metrics_calculator.print_metrics(head_names)

    return metrics_calculator


if __name__ == "__main__":
    demonstrate_correct_validation()