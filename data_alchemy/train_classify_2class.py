# train.py
import csv
import datetime
import json
import logging
import os
import sys
from typing import Optional, List

import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from data_alchemy.gru_moe_model_2class import ThreeLayerMoEWithSmartRouting
from data_alchemy.loss import create_fixed_class_weights, MultiHeadBinaryFocalLoss
from data_loader.data_loader import *  # 你已实现


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
    start_time = datetime.datetime.strptime(f"{start_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    end_date = sys.argv[5]
    end_time = datetime.datetime.strptime(f"{end_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    start_train(config, data_dir, market, start_time, end_time)

def test_data(config, data_dir, market, start_time, end_time):
    train_cfg = config["training"]
    model_cfg = config['model']
    device = get_device()
    print(f"使用设备: {device}")
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
                               mid_freq_type, low_freq_type, labels)  # TimeSeriesDataset(data_dir, market, val_start, val_end)

    # 测试集（仅最后评估一次）
    test_dataset = get_data_set(data_dir, "spot", "labels", market, test_start, test_end, interval,
                                seq_len, mid_freq_type, low_freq_type, labels)  # TimeSeriesDataset(data_dir, market, test_start, test_end)

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


def get_data_set(data_dir, biz, data_type, market, start_time, end_time, interval, seq_len, mid_type, low_type, labels):
    all_list = get_all_file_list(data_dir, biz, data_type, market, start_time, end_time, interval, seq_len=seq_len)
    return SegmentSets(all_list, data_dir, market, label_type=data_type, mid_type=mid_type, low_type=low_type, required_labels=labels)


from typing import Dict
import torch

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
        ls_classes = 5  #class_config['ls_choice'][scale]
        ls_logits = logits[:, current_offset:current_offset + ls_classes]
        ls_pred = ls_logits.argmax(dim=1)
        ls_label = labels[:, i]

        # Accuracy
        #acc_ls = (ls_pred == ls_label).float().mean().item()
        #result_dict[f"{scale}_ls_acc"] = round(acc_ls, 6)

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


def start_train(config, data_dir, market, start_time, end_time, resume_from: Optional[str] = None):
    train_cfg = config["training"]
    model_cfg = config['model']
    device = get_device()
    print(f"使用设备: {device}")
    total_duration = end_time - start_time

    save_dir = config["callbacks"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")
    csv_file = os.path.join(save_dir, "metrics.csv")
    test_result_file = os.path.join(save_dir, "test_metrics.json")

    # 推荐：固定验证/测试时长（更合理），或按比例
    val_ratio = train_cfg.get('val_ratio', 0.04)
    test_ratio = train_cfg.get('test_ratio', 0.012)

    low_freq_type = train_cfg.get("low_freq_type", "factor_k1h")
    mid_freq_type = train_cfg.get("mid_freq_type", "factor_k5m")
    label_type = train_cfg.get('label_type', 'ls0_labels')
    alphas = train_cfg.get("alphas")
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
    criterion = MultiHeadBinaryFocalLoss(alphas=alphas, gamma=2.0)

    prefetch = train_cfg.get('prefetch_factor', 1)
    if prefetch == 0:
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

    label_cols = all_cols[label_start:label_end]
    train_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, train_start, train_end,
                                 interval, seq_len, mid_freq_type, low_freq_type, label_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch)

    # 验证集（用于早停和调参）
    print(f"Val data: {val_start}-{train_end}, interval: {interval}")
    val_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, val_start, val_end, interval,
                               seq_len, mid_freq_type, low_freq_type, label_cols) #TimeSeriesDataset(data_dir, market, val_start, val_end)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch)

    # 测试集（仅最后评估一次）
    print(f"Test data: {test_start}-{test_end} seq_len: {seq_len}")
    test_dataset = get_data_set(data_dir, "spot", "ls1_labels", market, test_start, test_end, interval,
                                seq_len, mid_freq_type, low_freq_type, label_cols) # TimeSeriesDataset(data_dir, market, test_start, test_end)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch_factor=prefetch)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    # === 初始化 CSV ===

    # === 初始化 CSV ===
    labels = ['long_1min', 'short_1min','long_3min','short_3min', 'long_5min',
              'short_5min', "long_15min", 'short_15min', "long_30min", 'short_30min']
    selected_labels = labels[label_start:label_end]
    fieldnames = ["epoch", "train_loss", "val_loss"]
    log_indies = []
    for scale in selected_labels:
        scale_index = [f"{scale}_prec", f"{scale}_reca", f"{scale}_f1"]
        log_indies.append(scale_index)
        fieldnames += scale_index

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    # === 随机种子 ===
    torch.manual_seed(config["training"]["seed"])
    model = ThreeLayerMoEWithSmartRouting(
        low_input_dim=config["model"]["low_freq_dim"],
        mid_input_dim=config["model"]["mid_freq_dim"],
        high_input_dim=config["model"]["high_freq_dim"],
        class_config=config["class_config"],
        start_scale=label_start,
        end_scale=label_end,
        low_hidden=config["model"]["router_hidden"],
        low_layers=config["model"]["router_layers"],
        low_parallel=config["model"]["router_parallelism"],
        mid_hidden=config["model"]["expert_hidden"],
        mid_layers=config["model"]["expert_layers"],
        mid_parallel=config["model"]["expert_parallelism"],
        num_mid_experts=config["model"]["num_experts"],
        high_hidden=config["model"]["fusion_hidden"],
        high_layers=config["model"]["fusion_layers"],
        high_parallel=config["model"]["fusion_parallelism"],
        head_hidden=config["model"]["head_hidden"]
    ).to(device)
    # Model
    # === 优化器 & 调度器 ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    best_val_loss = float("inf")
    patience_counter = 0
    epochs = config["training"]["num_epochs"]

    if resume_from and os.path.exists(resume_from):
        logger.info(f".Resume training from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        logger.info(f".Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
    else:
        start_epoch = 0

    epochs = config["training"]["num_epochs"]
    grad_clip = config["training"]["grad_clip"]
    class_config = config["class_config"]
    class_weights = create_fixed_class_weights()
    class_weights = {k: v.to(device) for k, v in class_weights.items()}
    total_batchs = (len(train_dataset) + batch_size - 1) // batch_size

    break_on_debug = True
    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        cur_batch = 0
        start_time = datetime.datetime.now()
        model.update_epoch(epoch)
        for batch in train_loader:
            x_high, x_mid, x_low, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            if epoch < 5:
                logits = model(x_low, x_mid, x_high, return_regularization=False)
                reg_loss = 0
            else:
                # 从第5个epoch开始检查是否需要正则化
                logits, reg_loss = model(x_low, x_mid, x_high, return_regularization=True)
            loss = criterion(logits, labels)  # multi_scale_classification_loss(logits, labels, scale_names=scale_names, class_weights=class_weights)
            loss += reg_loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_train_loss += loss.item()
            cur_batch += 1
            end_time = datetime.datetime.now()
            print(f"[{cur_batch}/{total_batchs}] cost: {end_time - start_time}")
            start_time = end_time

            if break_on_debug:
                logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                break

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        total_val_loss = 0.0
        all_val_details = {}
        #collected_pred_details = defaultdict(list)
        with torch.no_grad():
            for batch in val_loader:
                x_high, x_mid, x_low, labels = [b.to(device) for b in batch]
                logits = model(x_low, x_mid, x_high, return_regularization=False)
                loss = criterion(logits, labels)
                details = criterion.compute_metrics_per_head(logits, labels)
                total_val_loss += loss.item()
                # 累积 loss 和 accuracy（保持你原有逻辑）
                for i in range(len(log_indies)):
                    scale_names = log_indies[i]
                    scale_detail = details[i]
                    for name,detail in zip(scale_names, scale_detail):
                        all_val_details[name] = all_val_details.get(name, 0) + detail

                if break_on_debug:
                    logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                    break

        avg_val_loss = total_val_loss / len(val_loader)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(avg_val_loss, 6),
            **{k: v / len(test_loader) for k, v in all_val_details.items()},
        }

        # 写入 CSV（确保 fieldnames 包含所有 _pred_clsX / _true_clsX）
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(log_dict)

        logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, "best_model.pth"))
            logger.info("→ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config["callbacks"]["patience"]:
                logger.info("Early stopping triggered.")
                break

        # --- Check for manual stop signal ---
        if os.path.exists(os.path.join(save_dir, "epoch.stop")):
            logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
            os.remove(os.path.join(save_dir, "epoch.stop"))  # 可选：自动清理
            break

    # ==========================================
    # 🔚 训练结束 → 加载最佳模型并在 test 集评估
    # ==========================================
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_test_loss = 0.0
    all_test_details = {}
    all_test_accuracies = {}

    with torch.no_grad():
        for batch in test_loader:
            x_high, x_mid, x_low, labels = [b.to(device) for b in batch]
            logits = model(x_low, x_mid, x_high, return_regularization = False)
            loss = criterion(logits, labels)
            total_test_loss += loss.item()
            details = criterion.compute_metrics_per_head(logits, labels)

            # 累积 loss
            for i in range(len(log_indies)):
                scale_names = log_indies[i]
                scale_detail = details[i]
                for name, detail in zip(scale_names, scale_detail):
                    all_val_details[name] = all_val_details.get(name, 0) + detail

            if break_on_debug:
                logger.info("Detected 'epoch.stop' file. Stopping training loop gracefully.")
                break
    # 平均
    avg_test_loss = total_test_loss / len(test_loader)
    for k in all_test_details:
        all_test_details[k] /= len(test_loader)
    for k in all_test_accuracies:
        all_test_accuracies[k] /= len(test_loader)

    # 构建最终结果
    test_metrics = {
        "test_loss": round(avg_test_loss, 6),
        "subtask_losses": {k: round(v, 6) for k, v in all_test_details.items()},
    }

    # 保存 JSON
    with open(test_result_file, "w", encoding="utf-8") as f:
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

    return test_metrics


if __name__ == "__main__":
    pass
    #main()

test_train()