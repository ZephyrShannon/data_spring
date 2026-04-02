def analyze_high_loss_reasons():
    """
    分析 loss > 300 的原因
    """

    print("\n" + "=" * 70)
    print("Loss > 300 的原因分析")
    print("=" * 70)

    # 模拟极端情况
    print("\n1. BCE Loss 本身可以非常大:")
    print("-" * 50)

    # 当预测概率接近 0 但真实标签为 1 时
    probs = torch.tensor([0.001, 0.0001, 0.00001])
    targets = torch.ones_like(probs)

    bce_losses = -torch.log(probs)
    for p, loss in zip(probs, bce_losses):
        print(f"  预测概率={p:.5f}, 真实=1 → BCE={loss:.2f}")

    print("\n2. Focal Weight 放大效果:")
    print("-" * 50)

    gamma = 2.0
    pt = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2])
    focal_weights = (1 - pt) ** gamma

    for p, fw in zip(pt, focal_weights):
        print(f"  pt={p:.3f}, focal_weight={fw:.4f} (放大 {fw:.1f} 倍)")

    print("\n3. Alpha Weight 放大效果:")
    print("-" * 50)

    alpha = 0.9
    pos_weight = alpha / (1 - alpha)  # 这里要小心！
    print(f"  alpha={alpha} → 正样本权重={pos_weight:.1f}")

    print("\n4. 组合效果示例:")
    print("-" * 50)

    # 极端情况：模型预测极差
    logits = torch.tensor([-10.0])  # 预测概率 ≈ 0.000045
    target = torch.tensor([1.0])

    probs = torch.sigmoid(logits)
    bce = -torch.log(probs)
    pt = probs
    focal_weight = (1 - pt) ** 2
    alpha_weight = 0.9

    total_loss = focal_weight * alpha_weight * bce

    print(f"  预测概率: {probs.item():.6f}")
    print(f"  BCE Loss: {bce.item():.2f}")
    print(f"  Focal Weight: {focal_weight.item():.4f}")
    print(f"  Alpha Weight: {alpha_weight}")
    print(f"  最终 Loss: {total_loss.item():.2f}")

    print("\n结论: 当模型预测很差时，Loss 很容易超过 300！")


analyze_high_loss_reasons()