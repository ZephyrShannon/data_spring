# train.py
import csv
import datetime
import gc
import json
import logging
import os
import sys
from typing import Optional, List
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_alchemy.loss import create_fixed_class_weights, MultiHeadBinaryFocalLoss
from data_alchemy.optimize_utils.find_learning_rate import find_best_lr, get_best_lr
from data_alchemy.optimize_utils.nan_checker import check_gradient_explosion
from data_alchemy.simple_stack_model import MultiFreqMultiLabelClassifier
from data_alchemy.test_metrics import ValidationMetrics
from data_loader.data_loader import *  # 你已实现
from data_alchemy.optimize_utils import nan_checker


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.vstack(all_preds)  # (N, 12)
    all_targets = np.vstack(all_targets)  # (N, 12)

    metrics = {}
    for i in range(all_preds.shape[1]):
        mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
        rmse = np.sqrt(mean_squared_error(all_targets[:, i], all_preds[:, i]))
        r2 = r2_score(all_targets[:, i], all_preds[:, i])
        metrics[f'target_{i}'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    # 全局平均
    avg_mae = np.mean([v['MAE'] for v in metrics.values()])
    avg_rmse = np.mean([v['RMSE'] for v in metrics.values()])
    avg_r2 = np.mean([v['R2'] for v in metrics.values()])
    metrics['average'] = {'MAE': avg_mae, 'RMSE': avg_rmse, 'R2': avg_r2}

    return metrics


def export_to_onnx(model, dummy_input, filepath="model.onnx"):
    torch.onnx.export(
        model,
        dummy_input,
        filepath,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ ONNX 模型已导出至: {filepath}")


def main():
    config_file = sys.argv[1]
    config = load_config(config_file)
    data_dir = sys.argv[2]
    market = sys.argv[3]
    start_date = sys.argv[4]
    if len(start_date) == 10:
        start_time = datetime.datetime.strptime(f"{start_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    else:
        start_time = datetime.datetime.strptime(f"{start_date}:00:00+0000", '%Y-%m-%dT%H:%M:%S%z')
    end_date = sys.argv[5]
    if len(end_date) == 10:
        end_time = datetime.datetime.strptime(f"{end_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    else:
        end_time = datetime.datetime.strptime(f"{end_date}:00:00+0000", '%Y-%m-%dT%H:%M:%S%z')
    if len(sys.argv) > 6:
        estimate = sys.argv[6].lower() == 'true'
    else:
        estimate = False
    start_train(config, data_dir, market, start_time, end_time, estimate=estimate)


def test_data(config, data_dir, market, start_time, end_time):
    train_cfg = config["training"]
    model_cfg = config['model']
    device = get_device()
    print(f"Use device: {device}")
    total_duration = end_time - start_time

    save_dir = config["callbacks"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")
    csv_file = os.path.join(save_dir, "metrics.csv")
    test_result_file = os.path.join(save_dir, "test_metrics.json")

    # 推荐：固定验证/测试时长（更合理），或按比例
    val_ratio = train_cfg.get('val_ratio', 0.01)
    test_ratio = train_cfg.get('test_ratio', 0.012)
    low_freq_type = train_cfg.get("low_freq_type", "factor_k1h")
    mid_freq_type = train_cfg.get("mid_freq_type", "factor_k5m")
    total_duration = end_time - start_time

    val_duration = total_duration * val_ratio

    test_duration = total_duration * test_ratio

    val_duration = datetime.timedelta(seconds=int((val_duration.total_seconds()) // 3600) * 3600)
    test_duration = datetime.timedelta(seconds=int((test_duration.total_seconds()) // 3600) * 3600)
    max_duration = datetime.timedelta(days=7)
    min_duration = datetime.timedelta(hours=1)
    if val_duration > max_duration:
        val_duration = max_duration
    if val_duration < min_duration:
        val_duration = min_duration
    if test_duration > max_duration:
        test_duration = max_duration
    if test_duration < min_duration:
        test_duration = min_duration

    test_start = end_time - test_duration
    val_start = test_start - val_duration

    train_start = start_time
    train_end = val_start
    val_end = test_start
    test_end = end_time
    interval = train_cfg.get('interval', 60)
    seq_len = model_cfg['seq_len']

    prefetch = train_cfg.get('prefetch_factor', 1)
    if prefetch == 0:
        prefetch = None
        num_workers = 0
    else:
        num_workers = 1
    labels = ["ls_choice_5m", "ls_choice_15m", "ls_choice_30m", "ls_choice_60m", "ls_choice_180m"]
    # 训练集
    train_dataset = get_data_set(data_dir, "spot", "ticks", market, train_start, train_end, interval,
                                 seq_len, mid_freq_type, low_freq_type, labels)

    # 验证集（用于早停和调参）
    val_dataset = get_data_set(data_dir, "spot", "ticks", market, val_start, val_end, interval, seq_len,
                               mid_freq_type, low_freq_type,
                               labels)  # TimeSeriesDataset(data_dir, market, val_start, val_end)

    # 测试集（仅最后评估一次）
    test_dataset = get_data_set(data_dir, "spot", "labels", market, test_start, test_end, interval,
                                seq_len, mid_freq_type, low_freq_type,
                                labels)  # TimeSeriesDataset(data_dir, market, test_start, test_end)

    print(f"Train dataset total length: {len(train_dataset)}\n")
    for i in range(len(train_dataset)):
        if i % 1000 == 0:
            print(f"Test: {i}")
        dt = train_dataset.get_item_datetime(i)
        try:
            train_dataset[i]
        except Exception as e:
            print(f"Try fetch data:[{dt}] failed")

    print(f"Val_dataset total length: {len(val_dataset)}\n")
    for i in range(len(val_dataset)):
        dt = val_dataset.get_item_datetime(i)
        try:
            val_dataset[i]
        except Exception as e:
            print(f"Try fetch data:[{dt}] failed")

    print(f"test_dataset total length: {len(test_dataset)}\n")
    for i in range(len(test_dataset)):
        dt = test_dataset.get_item_datetime(i)
        try:
            test_dataset[i]
        except Exception as e:
            print(f"Try fetch data:[{dt}] failed")


def test_train():
    config = load_config('configs/model.yaml')
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    start_date = "2024-01-01"
    format = '%Y-%m-%d %H:%M:%S%z'
    start_time = datetime.datetime.strptime(f"{start_date} 01:00:00+0000", format)
    end_date = "2024-01-01"
    end_time = datetime.datetime.strptime(f"{end_date} 23:00:00+0000", format)
    resume_from = "/Users/zephyr/codes/alpha_spring/data_spring/data_alchemy/checkpoints/exp_m4_v1/best_model.pth"
    start_train(config, data_dir, market, start_time, end_time, resume_from)


def get_data_set(data_dir, biz, lb_data_type, market, start_time, end_time, interval, md_interval, seq_len, mid_type, low_type,
                 labels, hf_data_type=None):
    all_list = get_all_file_list(data_dir, biz, lb_data_type, market, start_time, end_time, interval, seq_len=seq_len)
    md_num_per_epoch = int(interval / md_interval)
    return SegmentSets(all_list, data_dir, market, label_type=lb_data_type, md_data_interval=md_interval,  mid_type=mid_type, low_type=low_type, md_num_per_epoch=md_num_per_epoch,
                       required_labels=labels, hf_data_type=hf_data_type)


from typing import Dict
import torch
import tracemalloc


def calculate_accuracy_precision_recall_per_class(
        logits: torch.Tensor,
        labels: torch.Tensor,
        scales: List[str],
) -> Dict[str, float]:
    """
    计算每个子任务的准确率 + 每个类别的精确率和召回率
    输出格式:
      - {scale}_{task}_acc
      - {scale}_{task}_precision_cls{cls}
      - {scale}_{task}_recall_cls{cls}
    """
    current_offset = 0
    result_dict = {}

    for i, scale in enumerate(scales):
        # ===== LS Choice =====
        ls_classes = 5  # class_config['ls_choice'][scale]
        ls_logits = logits[:, current_offset:current_offset + ls_classes]
        ls_pred = ls_logits.argmax(dim=1)
        ls_label = labels[:, i]

        # Accuracy
        # acc_ls = (ls_pred == ls_label).float().mean().item()
        # result_dict[f"{scale}_ls_acc"] = round(acc_ls, 6)

        # Per-class Precision & Recall
        for cls in range(ls_classes):
            # 预测为 cls 的样本
            pred_mask = (ls_pred == cls)
            num_pred = pred_mask.sum().item()

            # 真实为 cls 的样本
            gt_mask = (ls_label == cls)
            num_gt = gt_mask.sum().item()

            # TP: 预测为 cls 且真实为 cls
            tp = (pred_mask & gt_mask).sum().item()

            # Precision = TP / (TP + FP) = TP / num_pred
            if num_pred > 0:
                precision_cls = tp / num_pred
            else:
                precision_cls = 0.0  # 或 float('nan')，但 0 更便于记录

            # Recall = TP / (TP + FN) = TP / num_gt
            if num_gt > 0:
                recall_cls = tp / num_gt
            else:
                recall_cls = 0.0

            result_dict[f"lb_{scale}_{cls}_prec"] = round(precision_cls, 6)
            result_dict[f"lb_{scale}_{cls}_reca"] = round(recall_cls, 6)

        current_offset += ls_classes

        # ===== Volatility =====
        '''
        vol_classes = class_config['volat'][scale]
        vol_logits = logits[:, current_offset:current_offset + vol_classes]
        vol_pred = vol_logits.argmax(dim=1)
        vol_label = labels[:, 2 * i + 1]

        # Accuracy
        #acc_vol = (vol_pred == vol_label).float().mean().item()
        #result_dict[f"{scale}_vol_acc"] = round(acc_vol, 6)

        # Per-class Precision & Recall
        for cls in range(vol_classes):
            pred_mask = (vol_pred == cls)
            num_pred = pred_mask.sum().item()

            gt_mask = (vol_label == cls)
            num_gt = gt_mask.sum().item()

            tp = (pred_mask & gt_mask).sum().item()

            precision_cls = tp / num_pred if num_pred > 0 else 0.0
            recall_cls = tp / num_gt if num_gt > 0 else 0.0

            result_dict[f"vol_{scale}_{cls}_prec"] = round(precision_cls, 6)
            result_dict[f"vol_{scale}_{cls}_reca"] = round(recall_cls, 6)

        current_offset += vol_classes
        '''

    return result_dict


class MemTracker:
    def __init__(self, name: str, snapshot=None):
        if snapshot is None:
            gc.collect()
            #snapshot = tracemalloc.take_snapshot()
        self.name = name
        self.snapshot = snapshot

    def record(self, hints: str, logger):
        gc.collect()
        '''
        snapshot_now = tracemalloc.take_snapshot()
        top_stats = snapshot_now.compare_to(self.snapshot, 'traceback')
        logger.info(f"MemTracker[{self.name}][{hints}]:")
        for stat in top_stats[:10]:
            trace_lines = stat.traceback.format()
            if any('/data_spring/' in line for line in trace_lines):
                logger.info(f"{stat}")
                tracebacks = '\n'.join(stat.traceback.format())
                logger.info(f"{tracebacks}")
        self.snapshot = snapshot_now
        '''


def check_model_gradients_by_component(model, logger):
    """
    按组件检查梯度（针对你的模型结构）
    """
    components = {
        'lf_extractor.dnn1': [],
        'lf_extractor.cnn': [],
        'lf_extractor.dnn2': [],
        'lf_extractor.rnn': [],
        'lf_extractor.output_head': [],
        'mf_extractor.dnn1': [],
        'mf_extractor.cnn': [],
        'mf_extractor.dnn2': [],
        'mf_extractor.rnn': [],
        'mf_extractor.output_head': [],
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.norm(2).item()

        # 根据参数名分类
        if 'lf_extractor.dnn1' in name:
            components['lf_extractor.dnn1'].append(grad_norm)
        elif 'lf_extractor.cnn' in name:
            components['lf_extractor.cnn'].append(grad_norm)
        elif 'lf_extractor.dnn2' in name:
            components['lf_extractor.dnn2'].append(grad_norm)
        elif 'lf_extractor.rnn' in name:
            components['lf_extractor.rnn'].append(grad_norm)
        elif 'lf_extractor.output_head' in name:
            components['lf_extractor.output_head'].append(grad_norm)
        if 'mf_extractor.dnn1' in name:
            components['mf_extractor.dnn1'].append(grad_norm)
        elif 'mf_extractor.cnn' in name:
            components['mf_extractor.cnn'].append(grad_norm)
        elif 'mf_extractor.dnn2' in name:
            components['mf_extractor.dnn2'].append(grad_norm)
        elif 'mf_extractor.rnn' in name:
            components['mf_extractor.rnn'].append(grad_norm)
        elif 'mf_extractor.output_head' in name:
            components['mf_extractor.output_head'].append(grad_norm)

    logger.info(f"\n=== {model.name}各组件梯度统计 ===")
    logger.info("组件      | 平均梯度   | 最小梯度   | 最大梯度   | 层数")
    logger.info("-" * 60)

    for comp_name, grads in components.items():
        if grads:
            avg_grad = sum(grads) / len(grads)
            min_grad = min(grads)
            max_grad = max(grads)
            logger.info(f"{comp_name:10s} | {avg_grad:.6f} | {min_grad:.6f} | {max_grad:.6f} | {len(grads):3d}")
    return components

def save_model_and_date(model, optimizer, recommended_lr, output_dir, index:int, start_date: datetime.datetime, end_date: datetime.datetime):
    save_file = f"{output_dir}/{index:04}_{start_date.year}{start_date.month:02}{start_date.day:02}{start_date.hour:02}{start_date.minute:02}_{end_date.year:02}{end_date.month:02}{end_date.day:02}{end_date.hour:02}{end_date.minute}.pth"
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr': recommended_lr,
    }, save_file)


def start_train(config, data_dir, market, start_time, end_time, resume_from: Optional[str] = None, estimate=False):
    train_cfg = config["training"]
    model_cfg = config['model']

    tracemalloc.start(25)
    procedure_tracker = MemTracker("procedure")
    overall_tracker = MemTracker("overall", procedure_tracker.snapshot)
    device = get_device()
    print(f"Use device: {device}")
    total_duration = end_time - start_time

    save_dir = config["callbacks"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")
    csv_file = os.path.join(save_dir, "metrics.csv")
    test_result_file = os.path.join(save_dir, "test_metrics.json")

    # 推荐：固定验证/测试时长（更合理），或按比例
    val_ratio = train_cfg.get('val_ratio', 0.04)
    test_ratio = train_cfg.get('test_ratio', 0.012)
    print(f"Val ratio: {val_ratio}, test_ratio: {test_ratio}")

    low_freq_type = train_cfg.get("low_freq_type", "factor_k1h")
    mid_freq_type = train_cfg.get("mid_freq_type", "factor_k5m")
    label_type = train_cfg.get('label_type', 'ls0_labels')
    alphas = train_cfg.get("alphas")
    total_duration = end_time - start_time

    val_duration = total_duration * val_ratio
    test_duration = total_duration * test_ratio
    val_duration = datetime.timedelta(seconds=int((val_duration.total_seconds()) // 3600) * 3600)
    test_duration = datetime.timedelta(seconds=int((test_duration.total_seconds()) // 3600) * 3600)
    max_duration = datetime.timedelta(days=30)
    min_duration = datetime.timedelta(hours=2)
    if val_duration > max_duration:
        val_duration = max_duration
    if val_duration < min_duration:
        val_duration = min_duration
    if test_duration > max_duration:
        test_duration = max_duration
    if test_duration < min_duration:
        test_duration = min_duration

    test_start = end_time - test_duration
    val_start = test_start - val_duration

    train_start = start_time
    train_end = val_start
    val_end = test_start
    test_end = end_time
    interval = train_cfg.get('interval', 60)
    seq_len = model_cfg['low_freq']['input_seq_len']


    prefetch = train_cfg.get('prefetch_factor', 1)
    if prefetch == 0 or estimate:
        prefetch = None
        num_workers = 0
    else:
        num_workers = 1
    # 训练集
    batch_size = train_cfg['batch_size']
    print(f"Train data: {train_start}-{train_end} @ {batch_size}")
    label_start = config['training'].get('label_start', 0)
    label_end = config['training'].get('label_end', 2)
    label_num = label_end - label_start
    all_cols = []
    scales = ['1min', '3min', "5min", "15min", "30min"]
    for test_freq in scales:
        all_cols.append(f"long_signal_{test_freq}")
        all_cols.append(f"short_signal_{test_freq}")
    md_data_interval = config['training']['md_interval']
    label_cols = all_cols[label_start:label_end]
    train_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, train_start, train_end,
                                 interval, md_data_interval, seq_len, mid_freq_type, low_freq_type, label_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              prefetch_factor=prefetch)

    # 验证集（用于早停和调参）
    print(f"Val data: {val_start}-{train_end}, interval: {interval}")
    val_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, val_start, val_end, 300,md_data_interval,
                               seq_len, mid_freq_type, low_freq_type,
                               label_cols)  # TimeSeriesDataset(data_dir, market, val_start, val_end)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=num_workers,
                            prefetch_factor=prefetch)

    # 测试集（仅最后评估一次）
    print(f"Test data: {test_start}-{test_end} seq_len: {seq_len}")
    test_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, test_start, test_end, interval,md_data_interval,
                                seq_len, mid_freq_type, low_freq_type,
                                label_cols)  # TimeSeriesDataset(data_dir, market, test_start, test_end)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                             prefetch_factor=prefetch)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    # === 初始化 CSV ===

    # === 初始化 CSV ===
    labels = ['long_1min', 'short_1min', 'long_3min', 'short_3min', 'long_5min',
              'short_5min', "long_15min", 'short_15min', "long_30min", 'short_30min']
    selected_labels = labels[label_start:label_end]
    fieldnames = ["epoch", "train_loss", "val_loss", "total_norm", "negative_num", "negative_mean", "mid_num",
                  "mid_mean", "positive_num", "positive_mean"]
    log_indies = []
    for scale in selected_labels:
        scale_index = [f"{scale}_prec", f"{scale}_reca", f"{scale}_f1"]
        log_indies.append(scale_index)
        fieldnames += scale_index
    csv_existed = os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_existed:
            writer.writeheader()

    if estimate:
        from data_alchemy.utils import MultiScaleClassificationMetrics
        metrics = MultiScaleClassificationMetrics(label_num)

    # === 随机种子 ===
    torch.manual_seed(config["training"]["seed"])

    model = MultiFreqMultiLabelClassifier(**(config['model']))

    no_decay_param_names = []
    normal_param = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # LayerNorm 通常有 'weight' 和 'bias'
            if hasattr(module, 'weight'):
                no_decay_param_names.append(f"{name}.weight")
            if hasattr(module, 'bias') and module.bias is not None:
                no_decay_param_names.append(f"{name}.bias")
        elif hasattr(module, 'bias') and module.bias is not None:
            no_decay_param_names.append(f"{name}.bias")
        else:
            normal_param.append(f'{name}')

    # 去重（虽然一般不会重复）
    no_decay_param_names = list(set(no_decay_param_names))

    # 构建 optimizer groups
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n not in no_decay_param_names],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in model.named_parameters() if n in no_decay_param_names],
            "weight_decay": 0.0,
        },
    ]
    # Model
    lr = config["training"].get("lr", None)
    weight_decay = config["training"]["weight_decay"]
    betas = config["training"].get("betas",  (0.9, 0.999))
    # === 优化器 & 调度器 ===


    best_val_loss = float("inf")
    patience_counter = 0

    total_batchs = (len(train_dataset) + batch_size - 1) // batch_size

    if resume_from is None:
        resume_from = os.path.join(save_dir, "best_model.pth")

    criterion = MultiHeadBinaryFocalLoss(num_heads=label_num, device=device, alphas=alphas, gamma=2.0)

    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resume training from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        reuse_all = config["training"].get('reuse_all', None)
        if reuse_all is not None:
            lr = checkpoint.get('lr', lr)
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=lr, weight_decay=weight_decay,
                betas=config['training'].get('betas', betas),
                eps=1e-8
            )
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", -1) + 1
        else:
            start_epoch = 0
            optimizer = None

        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        model_loaded = True
        if estimate:
            start_epoch = 0
    else:
        if estimate:
            raise Exception("Model record not found! Estimating failed")
        start_epoch = 0
        optimizer = None

    if optimizer is None:
        if lr is None:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-7, weight_decay=1e-4)

            # 执行 LR 测试（只跑 1 个 epoch）
            lrs, losses = find_best_lr(model, train_loader, optimizer, criterion, total_batchs, output_html=f"{save_dir}/lr.html",
                                   device=device)

            steepest_lr, recommended_lr = get_best_lr(lrs, losses, 3, 1);
            logger.info(f"Find best learning rate: {recommended_lr}")
        else:
            recommended_lr = lr
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=recommended_lr, weight_decay=1e-4,
            betas=config['training'].get('betas',betas),
            eps=1e-8
        )
    else:
        recommended_lr = lr


    epochs = config["training"]["num_epochs"]
        # 推荐搭配 OneCycleLR
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs, eta_min=1e-6)

    grad_clip = config["training"]["grad_clip"]
    #class_config = config["class_config"]
    class_weights = create_fixed_class_weights()
    class_weights = {k: v.to(device) for k, v in class_weights.items()}

    break_on_debug = False
    #procedure_tracker.record("Start_Epoch", logger)
    logger.info(f"Start training, {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        #epoch_tracker = MemTracker("epoch_tracker", procedure_tracker.snapshot)
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        # total_reg_loss = 0.0
        cur_batch = 0
        start_time = datetime.datetime.now()
        # model.update_epoch(epoch)
        gamma = 1. + min(1.0, epoch / 5)
        criterion.gamma = gamma
        # train_dataset.epoch = epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"start epoch:{epoch}, lr={current_lr}")
        test_nan = config.get("test_nan", False)
        if not estimate:
            for batch in train_loader:
                x_mid, x_low, labels = [b.to(device) for b in batch]
                idx = cur_batch * batch_size
                end_idx = idx + batch_size - 1
                cur_date = train_dataset.get_item_datetime(idx)
                end_date = train_dataset.get_item_datetime(end_idx)
                if test_nan:
                    if torch.isnan(x_mid).any():
                        err_msg = f"❌ 中频[{cur_date}-{end_date}]输入有 NaN! 数量: {torch.isnan(x_mid).int().sum()}"
                        logger.error(err_msg)
                        raise Exception(err_msg)

                    if torch.isnan(x_low).any():
                        error_msg = f"❌ 低频[{cur_date}-{end_date}]输入有 NaN! 数量: {torch.isnan(x_low).int().sum()}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    if torch.isnan(labels).any():
                        error_msg = f"❌ 标签[{cur_date}-{end_date}]有 NaN! 数量: {torch.isnan(labels).int().sum()}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    if torch.isinf(x_mid).any():
                        error_msg = f"⚠️ 中频[{cur_date}-{end_date}]输入有 Inf! 比例: {torch.isinf(x_mid).float().mean():.2%}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    if torch.isinf(x_low).any():
                        error_msg = f"⚠️ 低频[{cur_date}-{end_date}]输入有 Inf! 比例: {torch.isinf(x_mid).float().mean():.2%}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                    if torch.isinf(labels).any():
                        error_msg = f"⚠️ 标签[{cur_date}-{end_date}]有 Inf! 比例: {torch.isinf(x_mid).float().mean():.2%}"
                        logger.error(error_msg)
                        raise Exception(error_msg)

                optimizer.zero_grad()
                logits = model(x_low, x_mid)
                loss = criterion(logits, labels)
                loss_val = loss.item()
                end_time = datetime.datetime.now()
                # epoch_tracker.record(f"Train[{epoch},{cur_batch}]", logger)

                cur_batch += 1
                total_train_loss += loss_val
                print(
                    f"[{cur_batch}/{total_batchs}]([{cur_date}-{end_date}])cost: {end_time - start_time}, loss = {loss_val}, mean_loss:{total_train_loss / cur_batch}")
                start_time = end_time
                if loss_val != loss_val:
                    save_model_and_date(model, optimizer, recommended_lr, save_dir, cur_batch, cur_date, end_date)
                    if torch.isnan(logits).any():
                        error_msg = f"⚠️ 预测[{cur_date}-{end_date}]结果中有 nan! 比例: {torch.isnan(logits).float().mean():.2%}"
                        logger.error(error_msg)
                    else:
                        error_msg = f"loss 值为 0"
                    check_gradient_explosion(model)
                    raise Exception(error_msg)

                optimizer.zero_grad()
                loss.backward()
                #if grad_clip > 0:
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                end_time = datetime.datetime.now()
                if break_on_debug:
                    logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                    break
                else:
                    if (cur_batch % 300 == 0) or ( cur_batch == 1):
                        check_model_gradients_by_component(model, logger)
                        model.print_gradient_decay_report(logger)
                        model.mf_extractor.dnn2.print_gradient_analysis(logger, "mf_extractor.dnn2")
                        model.mf_extractor.dnn1.print_gradient_analysis(logger, "mf_extractor.dnn1")
                        model.lf_extractor.dnn2.print_gradient_analysis(logger, "lf_extractor.dnn2")
                        model.lf_extractor.dnn1.print_gradient_analysis(logger, "lf_extractor.dnn1")


            avg_train_loss = total_train_loss / cur_batch
        else:
            avg_train_loss = 0

        #procedure_tracker.record(f"Train[{epoch}] ends.",logger)
        # --- Validate ---
        model.eval()
        total_val_loss = 0.0
        all_val_details = {}
        # collected_pred_details = defaultdict(list)

        with torch.no_grad():
            all_probs_list = []
            metrics_calculator = ValidationMetrics(label_num, device)
            for batch in val_loader:
                x_mid, x_low, labels = [b.to(device) for b in batch]
                logits = model(x_low, x_mid)
                loss = criterion(logits, labels)
                details = criterion.compute_metrics_per_head(logits, labels)
                total_val_loss += loss.item()
                # 累积 loss 和 accuracy（保持你原有逻辑）
                for i in range(len(log_indies)):
                    scale_names = log_indies[i]
                    scale_detail = details[i]
                    for name, detail in zip(scale_names, scale_detail):
                        all_val_details[name] = all_val_details.get(name, 0) + detail
                all_probs_list.append(torch.sigmoid(logits).cpu().numpy())  # 转概率

                metrics_calculator.update(logits, labels)
                    # 转换为 CPU numpy 数组
                if break_on_debug:
                    logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                    break

            metrics_calculator.print_metrics(label_cols, logger)
            all_probs = np.vstack(all_probs_list)  # (N, S)
            logger.info(f"Probs - min: {all_probs.min():.4f}, max: {all_probs.max():.4f}, "
                      f"mean: {all_probs.mean():.4f}, std: {all_probs.std():.4f}")
            less_02 = all_probs < 0.25
            num_less_02 = less_02.sum()
            if num_less_02 < 1:
                mean_less_02 = 0.0
            else:
                mean_less_02 = all_probs[less_02].mean()
            larger_08 = all_probs > 0.75
            num_larger_08 = larger_08.sum()
            if num_larger_08 < 1:
                mean_larger_08 = 0.0
            else:
                mean_larger_08 = all_probs[larger_08].mean()
            mid = ((all_probs >= 0.4) & (all_probs <= 0.6))
            num_mid = mid.sum()
            mean_mid = all_probs[mid].mean()

            logger.info(f"Probability distribution: [0-0.2]: [{mean_less_02},{num_less_02}], "
                        f"[0.2-0.8]: [{mean_mid},{num_mid}] "
                        f"[>0.8]: [{mean_larger_08},{num_larger_08}]")

        avg_val_loss = total_val_loss / len(val_loader)
        # 在每个验证步骤后调用
        scheduler.step(epoch)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.info(f"Gradient norm: {total_norm:.6f}")

        # 2. 梯度消失时：检查激活函数、初始化、添加残差连接
        if total_norm < 1e-4:
            # 检查是否有梯度消失
            logger.warn(f"Warning: Gradients are vanishing: {total_norm}")
            # 考虑：使用LeakyReLU替代ReLU，检查初始化，添加skip connection

        # 3. 梯度爆炸时：加强梯度裁剪
        if total_norm > 100:
            logger.warn(f"Warning: Gradient are exploding: {total_norm}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=30.0)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            "total_norm": round(total_norm, 6),
            "negative_num": round(num_less_02,1),
            "negative_mean": round(mean_less_02, 6),
            "mid_num": round(num_mid, 1),
            "mid_mean": round(mean_mid, 6),
            "positive_num": round(num_larger_08, 1),
            "positive_mean": round(mean_larger_08, 6),
            **{k: v / len(test_loader) for k, v in all_val_details.items()},
        }

        # 写入 CSV（确保 fieldnames 包含所有 _pred_clsX / _true_clsX）
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(log_dict)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        if estimate:
            break
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': avg_val_loss,
                'lr': recommended_lr,
            }, os.path.join(save_dir, "best_model.pth"))
            logger.info("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config["callbacks"]["patience"]:
                logger.info("Early stopping triggered.")
                break
        stop_file = os.path.join(save_dir, "epoch.stop")
        # --- Check for manual stop signal ---
        if os.path.exists(stop_file):
            logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
            os.remove(stop_file)  # 可选：自动清理
            break
        elif break_on_debug:
            break
        else:
            procedure_tracker.record(f"Validation[{epoch} ends", logger)

    # ==========================================
    # 🔚 训练结束 → 加载最佳模型并在 test 集评估
    # ==========================================
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=False
                            )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_test_loss = 0.0
    all_test_details = {}
    all_test_accuracies = {}

    with torch.no_grad():
        metrics_calculator = ValidationMetrics(label_num,  device)
        for batch in test_loader:
            x_mid, x_low, labels = [b.to(device) for b in batch]
            logits = model(x_low, x_mid)
            loss = criterion(logits, labels)
            total_test_loss += loss.item()
            metrics_calculator.update(logits, labels)

            if break_on_debug:
                logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                break
    # 平均
    avg_test_loss = total_test_loss / len(test_loader)

    metrics_calculator.print_metrics(label_cols, logger)
    # 构建最终结果
    test_metrics = {
        "test_loss": round(avg_test_loss, 6),
        "subtask_losses": {k: round(v, 6) for k, v in all_test_details.items()},
    }

    # 保存 JSON
    with open(test_result_file, "a", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    # 打印报告
    logger.info("\n" + "=" * 50)
    logger.info("🧪 FINAL TEST EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Overall Test Loss: {avg_test_loss:.6f}")
    logger.info("\n📊 Subtask Losses:")
    for k, v in all_test_details.items():
        logger.info(f"  {k}: {v:.6f}")
    logger.info("=" * 50)
    logger.info(f"Test results saved to: {test_result_file}")
    overall_tracker.record("Over all mem", logger)
    return test_metrics


if __name__ == "__main__":
    main()
    pass

# test_train()
