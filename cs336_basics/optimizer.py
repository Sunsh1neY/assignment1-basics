from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # 1. 取词表维度（最后一维）的最大值，keepdim=True 保持维度用于后续广播减法
    # inputs 形状: (..., vocab_size) -> max_logits 形状: (..., 1)
    max_logits = inputs.max(dim=-1, keepdim=True).values

    # 2. 对每个样本/每个位置的所有 logits 减去各自最大值 m（数值稳定化平移）
    # 平移后每行最大值严格为 0.0，其余值 <= 0.0，因此 exp(shifted) ∈ (0, 1]，彻底杜绝 float32 上溢
    shifted = inputs - max_logits

    # 3. 从平移后的 logits 中取出真实目标 token 对应的项（即 o_y - m）
    # 注意细节 1：真实语料常存为 uint16/int32，但 PyTorch 的 gather 强制要求 index 为 int64 (.long())
    # 注意细节 2：必须用 .squeeze(-1) 只压缩最后一维，不能无参 .squeeze()，否则 batch_size=1 时会误删第 0 维
    # targets: (...) -> unsqueeze(-1): (..., 1) -> gather -> (..., 1) -> squeeze(-1): (...)
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1).long()).squeeze(-1)

    # 4. 计算平移后的 LogSumExp: log(sum_a exp(o_a - m))
    # 因为 shifted 中至少有一个元素等于 0，exp(0) = 1，所以求和必然 >= 1.0，log 永远不会遇到 log(0) = -inf
    # log_sum_exp 形状: (...)
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))

    # 5. 根据公式 log(sum exp(o_a - m)) - (o_y - m) 得到每个位置的负对数似然损失，形状: (...)
    loss = log_sum_exp - target_logits

    # 6. 对所有前置维度（如 batch_size，或 batch_size × seq_len）取平均，返回标量 Tensor，形状: ()
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        # 1. 参数合法性校验，避免传入负学习率或超出 [0, 1) 的动量衰减系数导致发散
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 2. 构造默认超参数字典
        # 注意 betas 按元组存入，与 PyTorch 原生 Optimizer 约定对齐，方便学习率调度器直接读取/修改 group["lr"]
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        # 3. 交给父类 Optimizer 解析 params 并初始化 self.param_groups 和 self.state
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # 1. 如果调用方提供了 closure（需要重新前向+反向计算 loss 的场景），先临时恢复梯度追踪来执行它
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 2. 遍历每一个参数组（不同参数组可以独立配置 lr / weight_decay 等）
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 3. 遍历该参数组中的每一个可训练权重张量 p
            for p in group["params"]:
                # 如果某个参数当前没有梯度（比如冻结了，或者某次前向没经过它），直接跳过
                if p.grad is None:
                    continue

                # 取出当前参数的梯度 g
                grad = p.grad

                # 4. 读取该参数 p 在优化器里的专属状态字典
                state = self.state[p]
                # 首次调用 step() 时，延迟初始化步数 t 与一阶/二阶矩缓冲（形状、dtype、设备与 p 完全一致）
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # 5. 更新迭代计数 t（必须从 1 开始，否则 t=0 时 1 - beta^0 = 0 会除零）
                t = state["t"] + 1
                state["t"] = t

                # 取出缓冲区引用（后续用原地操作直接改写这两块显存/内存）
                m = state["m"]
                v = state["v"]

                # 6. 计算偏差修正后的有效学习率 alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)（对应 Algorithm 1 第 7 行）
                # 原因：m 和 v 都从 0 开始，训练早期严重偏小，偏差修正可以抵消冷启动误差
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # 7. 先执行解耦权重衰减 (Decoupled Weight Decay): theta <- theta - lr * weight_decay * theta（对应 Algorithm 1 第 8 行）
                # 关键顺序：必须在动量梯度更新之前衰减当前权重 theta！
                # 否则如果先更新再衰减，相当于把自适应梯度步长 alpha_t * m / (sqrt(v)+eps) 也误乘了一次 (1 - lr * weight_decay)。
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # 8. 更新一阶矩估计 (梯度均值 EMA): m <- beta1 * m + (1 - beta1) * g（对应 Algorithm 1 第 9 行）
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 9. 更新二阶矩估计 (梯度平方 EMA): v <- beta2 * v + (1 - beta2) * g^2（对应 Algorithm 1 第 10 行）
                # addcmul_(a, b, value=c) 相当于原地加 c * (a * b)，省一次中间临时张量的内存分配
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 10. 再执行自适应梯度步长更新: theta <- theta - alpha_t * m / (sqrt(v) + eps)（对应 Algorithm 1 第 11 行）
                # 注意：分母是先对 v 开平方根再加 eps（不是 sqrt(v + eps)）
                denom = v.sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-alpha_t)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 阶段 1：线性预热 (Warmup)，当 0 <= it < warmup_iters 时
    # 原因：刚开训时 Adam 的二阶矩 v 尚未收集够统计量、权重也是随机的，直接用大学习率容易瞬间震荡发散
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    # 阶段 3（含边界保护）：当超过余弦周期 T_c，或者特殊情况 T_c == T_w（避免后面除以 0）时
    # 停留在设定的最小学习率 min_learning_rate 继续微调收敛
    if it > cosine_cycle_iters or cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    # 阶段 2：余弦退火 (Cosine Annealing)，当 warmup_iters <= it <= cosine_cycle_iters 时
    # 计算在退火阶段中的归一化进度 progress ∈ [0.0, 1.0]
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

    # 利用 cos(0)=1 -> cos(pi)=-1 的性质，构造从 1.0 平滑下降到 0.0 的曲线因子
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    # 在 [min_learning_rate, max_learning_rate] 之间做平滑插值
    return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)



def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    # 1. 一次性收集所有非空梯度到列表中
    # 必须一次收集完：因为传入的 parameters 往往是 model.parameters() 生成器，
    # 如果先遍历一遍算范数、再遍历一遍做缩放，第二次循环时生成器早已耗尽！
    grads = [p.grad for p in parameters if p.grad is not None]

    # 2. 边界处理：如果所有参数都没有梯度（比如全部被冻结 requires_grad=False），直接返回
    # 防止下面对空列表调用 torch.stack([]) 抛出 RuntimeError
    if not grads:
        return

    # 3. 计算全局所有参数梯度拼接后的总 L2 范数：sqrt(sum_p sum(g_p^2))
    # 细节：用 detach() 脱离计算图，用 .float() 确保半精度(bf16/fp16)训练时累加平方也不会溢出，
    # 且每个分块都在自身设备上求和后 stack，兼容 CPU/GPU。
    total_norm_sq = torch.stack([g.detach().float().pow(2).sum() for g in grads]).sum()
    total_norm = torch.sqrt(total_norm_sq)

    # 4. 只有当全局总范数超过阈值 max_l2_norm 时才执行裁剪；未超限则保持梯度原封不动
    if total_norm > max_l2_norm:
        # 缩放到目标上界 M，分母加 eps=1e-6 防止数值极端情况除零
        scale = max_l2_norm / (total_norm + eps)

        # 5. 原地 (in-place) 对每个参数梯度乘上同一个缩放系数 scale
        with torch.no_grad():
            for g in grads:
                g.mul_(scale)