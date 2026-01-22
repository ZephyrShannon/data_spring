from typing import List

import torch


def calculate_precision(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算精确率 (Precision)
    Precision = TP / (TP + FP)

    Args:
        preds: 预测值 (0或1)
        labels: 真实标签 (0或1)

    Returns:
        precision: 精确率
    """
    # 确保是整数类型
    preds = preds.long()
    labels = labels.long()

    # 计算真正例 (True Positive)
    tp = ((preds == 1) & (labels == 1)).sum().float()

    # 计算预测为正例的总数
    predicted_positives = (preds == 1).sum().float()

    # 避免除零
    if predicted_positives == 0:
        return 0.0

    precision = tp / predicted_positives
    return precision.item()


def calculate_recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    计算召回率 (Recall)
    Recall = TP / (TP + FN)

    Args:
        preds: 预测值 (0或1)
        labels: 真实标签 (0或1)

    Returns:
        recall: 召回率
    """
    # 确保是整数类型
    preds = preds.long()
    labels = labels.long()

    # 计算真正例 (True Positive)
    tp = ((preds == 1) & (labels == 1)).sum().float()

    # 计算实际为正例的总数
    actual_positives = (labels == 1).sum().float()

    # 避免除零
    if actual_positives == 0:
        return 0.0

    recall = tp / actual_positives
    return recall.item()


class ClassificationMetrics:
    """完整的分类指标计算"""

    @staticmethod
    def calculate_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, int]:
        """
        计算混淆矩阵

        Returns:
            Dict with TP, TN, FP, FN counts
        """
        preds = preds.long()
        labels = labels.long()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        return {
            'tp': tp, 'tn': tn,
            'fp': fp, 'fn': fn,
            'total': tp + tn + fp + fn
        }

    @staticmethod
    def calculate_precision(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> float:
        """计算精确率"""
        conf_matrix = ClassificationMetrics.calculate_confusion_matrix(preds, labels)
        tp, fp = conf_matrix['tp'], conf_matrix['fp']

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp + eps)

    @staticmethod
    def calculate_recall(preds: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> float:
        """计算召回率"""
        conf_matrix = ClassificationMetrics.calculate_confusion_matrix(preds, labels)
        tp, fn = conf_matrix['tp'], conf_matrix['fn']

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn + eps)

    @staticmethod
    def calculate_f1_score(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """计算F1分数"""
        precision = ClassificationMetrics.calculate_precision(preds, labels)
        recall = ClassificationMetrics.calculate_recall(preds, labels)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def calculate_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """计算准确率"""
        conf_matrix = ClassificationMetrics.calculate_confusion_matrix(preds, labels)
        tp, tn, total = conf_matrix['tp'], conf_matrix['tn'], conf_matrix['total']

        return (tp + tn) / total if total > 0 else 0.0

    @staticmethod
    def calculate_all_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'precision': ClassificationMetrics.calculate_precision(preds, labels),
            'recall': ClassificationMetrics.calculate_recall(preds, labels),
            'f1': ClassificationMetrics.calculate_f1_score(preds, labels),
            'accuracy': ClassificationMetrics.calculate_accuracy(preds, labels),
            'confusion_matrix': ClassificationMetrics.calculate_confusion_matrix(preds, labels)
        }


class MultiScaleClassificationMetrics:
    """多时间尺度分类指标计算"""

    def __init__(self, num_scales: int):
        self.num_scales = num_scales

    def calculate_metrics_per_scale(self, all_probs: torch.Tensor, all_labels: torch.Tensor) -> List[Dict]:
        """
        为每个时间尺度分别计算指标

        Args:
            all_probs: (N, num_scales) 预测概率
            all_labels: (N, num_scales) 真实标签

        Returns:
            List of metric dicts for each scale
        """
        results = []

        for scale_idx in range(self.num_scales):
            scale_probs = all_probs[:, scale_idx]
            scale_labels = all_labels[:, scale_idx]

            # 使用默认阈值0.5
            scale_preds = (scale_probs > 0.5).float()

            metrics = ClassificationMetrics.calculate_all_metrics(scale_preds, scale_labels)

            # 添加一些额外信息
            metrics['scale_idx'] = scale_idx
            metrics['avg_prob'] = scale_probs.mean().item()
            metrics['label_distribution'] = {
                'positive': (scale_labels == 1).sum().item(),
                'negative': (scale_labels == 0).sum().item(),
                'total': len(scale_labels)
            }

            results.append(metrics)

        return results

    def find_optimal_thresholds(self, all_probs: torch.Tensor, all_labels: torch.Tensor,
                                target_metric: str = 'f1') -> List[float]:
        """
        为每个时间尺度寻找最优阈值

        Args:
            target_metric: 优化的目标指标 ('f1', 'precision', 'recall', 'balanced')
        """
        optimal_thresholds = []

        for scale_idx in range(self.num_scales):
            scale_probs = all_probs[:, scale_idx]
            scale_labels = all_labels[:, scale_idx]

            best_threshold = 0.5
            best_score = -1

            # 在多个阈值下评估
            for threshold in torch.arange(0.1, 0.9, 0.05):
                preds = (scale_probs > threshold).float()

                if target_metric == 'f1':
                    score = ClassificationMetrics.calculate_f1_score(preds, scale_labels)
                elif target_metric == 'precision':
                    score = ClassificationMetrics.calculate_precision(preds, scale_labels)
                elif target_metric == 'recall':
                    score = ClassificationMetrics.calculate_recall(preds, scale_labels)
                elif target_metric == 'balanced':
                    # 平衡精确率和召回率
                    precision = ClassificationMetrics.calculate_precision(preds, scale_labels)
                    recall = ClassificationMetrics.calculate_recall(preds, scale_labels)
                    score = (precision + recall) / 2

                if score > best_score:
                    best_score = score
                    best_threshold = threshold.item()

            optimal_thresholds.append(best_threshold)

        return optimal_thresholds


