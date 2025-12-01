# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import yaml
import os, sys
from lstm_moe_model import LSTMMoEModel
from data_loader.data_loader import *  # 你已实现
import datetime
import csv
import time

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
    train_cfg = config['training']
    model_cfg = config['model']
    data_dir = sys.argv[2],
    market = sys.argv[3]
    start_date = sys.argv[4]
    start_time = datetime.datetime.strptime(f"{start_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    end_date = sys.argv[5]
    end_time = datetime.datetime.strptime(f"{end_date} 00:00:00+0000", '%Y-%m-%d %H:%M:%S%z')
    start_train(train_cfg, model_cfg, data_dir, market, start_time, end_time)

def test_train():
    config = load_config('configs/model.yaml')
    train_cfg = config['training']
    model_cfg = config['model']
    data_dir = "/Users/zephyr/codes/alpha_spring/data_spring/data"
    market = "BTC_USDT"
    start_date = "2024-01-01"
    format = '%Y-%m-%d %H:%M:%S%z'
    start_time = datetime.datetime.strptime(f"{start_date} 01:00:00+0000", format)
    end_date = "2024-01-01"
    end_time = datetime.datetime.strptime(f"{end_date} 22:00:00+0000", format)
    start_train(train_cfg, model_cfg, data_dir, market, start_time, end_time)

'''
Add new segment:[2023-03-01:00-2023-03-10:16]
Add new segment:[2023-03-10:19-2023-04-16:03]
Add new segment:[2023-04-16:06-2023-06-10:03]
Add new segment:[2023-06-10:05-2023-08-13:02]
Add new segment:[2023-08-13:09-2023-12-11:00]
Add new segment:[2023-12-11:03-2024-01-02:12]
Add new segment:[2024-01-02:15-2024-02-20:14]
Add new segment:[2024-02-20:16-2024-02-22:12]
Add new segment:[2024-02-22:18-2024-04-15:01]
Add new segment:[2024-04-15:04-2024-09-26:15]
Add new segment:[2024-09-26:17-2024-12-04:18]
Add new segment:[2024-12-04:21-2025-03-03:21]
Add new segment:[2025-03-04:02-2025-06-08:20]
Add new segment:[2025-06-08:22-2025-10-13:15]
Add new segment:[2025-10-13:19-2025-10-31:22]
'''


def get_data_set(data_dir, biz, data_type, market, start_time, end_time, interval):
    all_list = get_all_file_list(data_dir, biz, data_type, market, start_time, end_time, interval)
    return SegmentSets(all_list)


def start_train(train_cfg, model_cfg, data_dir, market, start_time, end_time):
    device = get_device()
    print(f"使用设备: {device}")
    total_duration = end_time - start_time

    # 推荐：固定验证/测试时长（更合理），或按比例
    val_ratio = train_cfg.get('val_ratio', 0.01)
    test_ratio = train_cfg.get('test_ratio', 0.012)

    total_duration = end_time - start_time

    val_duration = total_duration * val_ratio
    test_duration = total_duration * test_ratio
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
    # 训练集
    train_dataset = get_data_set(data_dir, "spot", "ticks", market, train_start, train_end, interval)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=0)

    # 验证集（用于早停和调参）
    val_dataset = get_data_set(data_dir, "spot", "ticks", market, val_start, val_end, interval) #TimeSeriesDataset(data_dir, market, val_start, val_end)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=0)

    # 测试集（仅最后评估一次）
    test_dataset = get_data_set(data_dir, "spot", "labels", market, test_start, test_end, interval) # TimeSeriesDataset(data_dir, market, test_start, test_end)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=0)


    # Model
    model = LSTMMoEModel(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_cfg['lr_scheduler']['factor'],
        patience=train_cfg['lr_scheduler']['patience'],
        min_lr=train_cfg['lr_scheduler']['min_lr']
    )

    # === 日志文件准备 ===
    log_file = "training_log.csv"
    # 写入表头（如果文件不存在）
    write_header = not os.path.exists(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'best_val_loss'])
    # Training
    patience_counter = 0
    model.train()

    best_val_loss = float('inf')
    for epoch in range(train_cfg['num_epochs']):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            start_dt = datetime.datetime.now()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
            end_time = datetime.datetime.now()
            epoch_time  = end_time - start_dt
            print(f"Train one data Cost {epoch_time}")
            time.sleep(0.001)

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证（关键！）---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)

        # 早停基于验证损失
        if avg_val_loss < best_val_loss - train_cfg['early_stopping']['min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Epoch {epoch + 1}: 保存最佳模型 (Val Loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{current_lr:.2e}",
                f"{best_val_loss:.6f}"
            ])

        print(f"Epoch {epoch + 1}/{train_cfg['num_epochs']}, "
              f"Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}, "
              f"LR: {current_lr:.2e}")

        if patience_counter >= train_cfg['early_stopping']['patience']:
            print("🛑 早停触发")
            break

    # 加载最佳模型进行评估
    # 加载最佳模型，在 TEST SET 上评估（只做一次！）
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    metrics = evaluate_model(model, test_loader, device)  # ← 用 test_loader！


    print("\n📊 评估结果:")
    for target, met in metrics.items():
        if target == 'average':
            print(f"\n📈 平均指标:")
        else:
            print(f"\n🎯 {target}:")
        print(f"  MAE:  {met['MAE']:.6f}")
        print(f"  RMSE: {met['RMSE']:.6f}")
        print(f"  R²:   {met['R2']:.6f}")

    # 导出 ONNX
    model.eval()
    dummy_input = torch.randn(1, model_cfg['total_input_dim'], model_cfg['seq_len']).to(device)
    export_to_onnx(model, dummy_input, "lstm_moe_model.onnx")


if __name__ == "__main__":
    pass
    #main()