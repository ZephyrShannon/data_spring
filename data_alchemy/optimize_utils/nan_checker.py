import torch

def diagnose_nan_immediately(model, batch):
    """
    立即诊断 NaN 的来源
    """
    print("=" * 70)
    print("NaN 紧急诊断")
    print("=" * 70)

    # 检查输入数据
    print("\n1. 检查输入数据:")
    print("-" * 40)

    x_mid, x_low, labels = batch

    if torch.isnan(x_mid).any():
        print(f"❌ 中频输入有 NaN! 比例: {torch.isnan(x_mid).float().mean():.2%}")

    if torch.isnan(x_low).any():
        print(f"❌ 低频输入有 NaN! 比例: {torch.isnan(x_low).float().mean():.2%}")

    if torch.isnan(labels).any():
        print(f"❌ 标签有 NaN! 比例: {torch.isnan(labels).float().mean():.2%}")

    if torch.isinf(x_mid).any():
        print(f"⚠️ 中频输入有 Inf! 比例: {torch.isinf(x_mid).float().mean():.2%}")

    # 检查模型参数
    print("\n2. 检查模型参数:")
    print("-" * 40)

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ 参数有 NaN: {name}")

        if torch.isinf(param).any():
            print(f"⚠️ 参数有 Inf: {name}")

    # 检查模型前向传播
    print("\n3. 检查前向传播过程:")
    print("-" * 40)

    model.eval()

    with torch.no_grad():
        # 逐层检查
        def check_layer_outputs(module, input, output):
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"  ❌ {module.__class__.__name__} 输出有 NaN")
                if torch.isinf(output).any():
                    print(f"  ⚠️ {module.__class__.__name__} 输出有 Inf")
                if output.abs().max() > 1e6:
                    print(f"  ⚠️ {module.__class__.__name__} 输出过大: {output.abs().max():.2e}")

        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Sequential, nn.ModuleList)):
                hooks.append(module.register_forward_hook(check_layer_outputs))

        # 前向传播
        logits = model(x_low, x_mid)

        # 检查最终输出
        if torch.isnan(logits).any():
            print(f"❌ 模型输出有 NaN! 比例: {torch.isnan(logits).float().mean():.2%}")

        # 移除钩子
        for hook in hooks:
            hook.remove()

    return


def check_gradient_explosion(model):
    """
    检查梯度爆炸
    """
    print("\n" + "=" * 70)
    print("梯度爆炸检查")
    print("=" * 70)

    # 检查参数是否已变为 NaN
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)

    if nan_params:
        print(f"发现 {len(nan_params)} 个参数变为 NaN:")
        for name in nan_params[:10]:
            print(f"  ❌ {name}")

        print("\n解决方案:")
        print("1. 增加梯度裁剪强度")
        print("2. 降低学习率")
        print("3. 检查梯度是否有 NaN")

        return True

    # 检查梯度
    total_grad_norm = 0
    max_grad = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            max_grad = max(max_grad, grad_norm)

            if torch.isnan(param.grad).any():
                print(f"❌ 梯度有 NaN: {name}")
                return True

    total_grad_norm = total_grad_norm ** 0.5

    print(f"总梯度范数: {total_grad_norm:.6f}")
    print(f"最大梯度: {max_grad:.6f}")

    if total_grad_norm > 100:
        print("⚠️ 梯度爆炸！需要加强梯度裁剪")
        return True

    return False


def check_numerical_stability(model, logits, loss):
    """
    检查数值稳定性
    """
    print("\n" + "=" * 70)
    print("数值稳定性检查")
    print("=" * 70)

    # 检查 logits
    print("\n1. 检查 logits:")
    print(f"  范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"  均值: {logits.mean().item():.4f}")
    print(f"  标准差: {logits.std().item():.4f}")

    if logits.abs().max() > 50:
        print("⚠️ logits 过大，sigmoid 会饱和")
        print("  解决方案: 检查模型最后一层初始化")

    # 检查概率
    probs = torch.sigmoid(logits)
    print("\n2. 检查概率:")
    print(f"  范围: [{probs.min().item():.6f}, {probs.max().item():.6f}]")

    if probs.min() == 0:
        print("❌ 概率为0，会导致 log(0) 无穷大")

    if probs.max() == 1:
        print("❌ 概率为1，会导致 log(1-1)=log(0) 无穷大")

    # 检查损失
    print("\n3. 检查损失:")
    print(f"  loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")

    if torch.isnan(loss):
        print("❌ 损失是 NaN！")

    return


def check_learning_rate(optimizer, loss_history):
    """
    检查学习率是否太大
    """
    print("\n" + "=" * 70)
    print("学习率检查")
    print("=" * 70)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"当前学习率: {current_lr:.6f}")

    if len(loss_history) > 5:
        recent_losses = loss_history[-5:]
        if any(np.isnan(l) for l in recent_losses):
            print("❌ 最近出现 NaN，可能学习率太大")

            # 建议降低学习率
            new_lr = current_lr * 0.1
            print(f"\n建议降低学习率: {current_lr:.6f} -> {new_lr:.6f}")

            print("\n或者使用学习率 warmup:")
            print('''
            def warmup_lr(optimizer, step, warmup_steps):
                if step < warmup_steps:
                    lr = step / warmup_steps * base_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
            ''')