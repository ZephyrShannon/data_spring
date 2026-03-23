import torch
import numpy as np


def find_best_lr(
        model,
        train_loader,
        optimizer,
        criterion,
        total_batchs: int,
        init_lr=1e-7,
        final_lr=10,
        beta=0.98,
        output_html=None,  # ← 新增：HTML 输出路径
        device=None,
):
    """
    执行 Learning Rate Range Test，并可选输出交互式 HTML 图表。

    Args:
        model: 待测试的模型
        train_loader: 训练数据加载器（建议 shuffle=True，时间序列可设 False）
        optimizer: 优化器（如 AdamW），lr 会被覆盖
        criterion: 损失函数
        total_batchs: 总batch数每epoch
        init_lr: 起始学习率（默认 1e-7）
        final_lr: 结束学习率（默认 10）
        beta: loss 平滑系数（默认 0.98）
        output_html: str 或 None。若提供路径（如 'lr_test.html'），则保存 HTML 图表
        device: 设备（如 'cuda' 或 'cpu'），若为 None 则自动检测

    Returns:
        lrs (list): 学习率列表
        losses (list): 平滑后的损失列表:
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    num_batches = len(train_loader)

    # 初始化 lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr

    lrs = []
    losses = []
    avg_loss = 0.0
    best_loss = float('inf')

    for i, batch in enumerate(train_loader):
        # 动态计算当前 lr（指数增长）
        lr = init_lr * (final_lr / init_lr) ** (i / num_batches)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        x_mid, x_low, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits = model(x_low, x_mid)
        loss = criterion(logits, labels)
        loss_val = loss.item()

        loss.backward()
        optimizer.step()
        # 平滑 loss
        avg_loss = beta * avg_loss + (1 - beta) * loss_val
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))

        lrs.append(lr + 1e-12)
        losses.append(smoothed_loss)

        # 提前终止：loss 爆炸
        if smoothed_loss > 30:
            print(f"Loss exploded (smoothed={smoothed_loss:.4f}) at lr={lr:.2e}, stopping early.")
            break
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        print(
            f"[{i}/{total_batchs}]  loss = {loss_val}")

    # ======== 新增：生成 HTML 图表 ========
    if output_html is not None:
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo

            # 计算推荐 lr（最陡下降点）
            if len(lrs) > 10:
                grads = np.gradient(losses, np.log(lrs))  # 避免除零
                steepest_idx = int(np.argmin(grads[5:-5]) + 5)  # 忽略首尾噪声
                best_lr_approx = lrs[steepest_idx]
            else:
                best_lr_approx = lrs[np.argmin(losses)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lrs,
                y=losses,
                mode='lines+markers',
                name='Smoothed Loss',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=4)
            ))

            # 添加推荐 lr 线
            fig.add_vline(
                x=best_lr_approx,
                line=dict(color="red", dash="dash", width=2),
                annotation_text=f"Recommended LR ≈ {best_lr_approx:.2e}",
                annotation_position="top right"
            )

            fig.update_layout(
                title="Learning Rate Range Test",
                xaxis_title="Learning Rate (log scale)",
                yaxis_title="Smoothed Loss",
                xaxis_type="log",
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                template="plotly_white",
                hovermode="x unified",
                width=900,
                height=600
            )

            pyo.plot(fig, filename=output_html, auto_open=False)
            print(f"✅ LR Range Test chart saved to: {output_html}")
            print(f"💡 Recommended learning rate: {best_lr_approx:.2e}")

        except ImportError:
            print("⚠️ Warning: plotly not installed. Skipping HTML output.")
            print("   Run 'pip install plotly' to enable interactive charts.")

    return lrs, losses


def get_best_lr(lrs, losses, skip_begin=10, skip_end=5):
    # 跳过开头和结尾不稳定部分
    lrs = lrs[skip_begin:-skip_end]
    losses = losses[skip_begin:-skip_end]

    # 找最小 loss 对应的 lr
    min_idx = np.argmin(losses)
    best_lr = lrs[min_idx]

    # 更稳健：取下降最陡处（梯度最大负值）
    grads = np.gradient(losses, np.log(lrs))
    steepest_idx = np.argmin(grads)  # 最负梯度
    steepest_lr = lrs[steepest_idx]

    return steepest_lr, best_lr