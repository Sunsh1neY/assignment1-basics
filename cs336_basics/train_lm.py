from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

# 确保从任意工作目录运行本脚本时都能导入根目录下的 cs336_basics 包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.architecture import TransformerLM  # noqa: E402
from cs336_basics.optimizer import (  # noqa: E402
    AdamW,
    clip_gradients,
    cross_entropy,
    get_lr_cosine_schedule,
)
from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint  # noqa: E402


def load_tokens_memmap(path: Path, vocab_size: int) -> npt.NDArray:
    # 1. 根据文件后缀选择正确的零拷贝内存映射方式（PDF 第 35 页提示）
    # 关键避坑：用 np.save 生成的 .npy 开头带有 128 字节左右魔数文件头，必须用 np.load(..., mmap_mode="r") 解析；
    # 若误用裸 np.memmap 打开 .npy，会把文件头字节直接当成非法 token ID！
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.memmap(path, dtype=np.uint16, mode="r")

    # 2. 抽检数据前 10 万个 token 的最大 ID，防止训练前不小心配错了词表（比如拿 32K OWT 配 10K 模型）
    sample_max = int(arr[: min(100_000, len(arr))].max())
    if sample_max >= vocab_size:
        raise ValueError(
            f"数据集 {path} 中存在越界 token ID ({sample_max}) >= vocab_size ({vocab_size})，请检查词表配置"
        )
    return arr


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> tuple[float, float]:
    # 1. 切换到评估模式，并在 @torch.no_grad() 下省去激活值计算图内存
    model.eval()
    losses: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            logits = model(x)  # 形状: (batch_size, context_length, vocab_size)
            total += cross_entropy(logits, y).item()
        losses[split] = total / eval_iters

    # 2. 评估结束必须切回训练模式
    model.train()
    return losses["train"], losses["val"]


def log_eval_metrics(
    model: torch.nn.Module,
    train_data: npt.NDArray,
    val_data: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
    metrics_path: Path,
    step: int,
    t0: float,
    lr: float,
    tag: str = "eval",
) -> tuple[float, float, float]:
    """Run train/val loss evaluation, print summary, and append one JSONL row."""
    train_loss, val_loss = estimate_loss(
        model, train_data, val_data, batch_size, context_length, device, eval_iters
    )
    elapsed = time.perf_counter() - t0
    val_ppl = math.exp(min(val_loss, 20.0))  # 防止极早期 loss 过大时 exp 溢出
    print(
        f"[{tag}] step={step:5d} | time={elapsed:7.1f}s | lr={lr:.3e} "
        f"| train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
    )
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "wall_clock_sec": round(elapsed, 2),
                    "lr": lr,
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "val_ppl": round(val_ppl, 4),
                }
            )
            + "\n"
        )
    return train_loss, val_loss, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch (CS336 Section 5.3)")

    # 数据与运行环境参数
    parser.add_argument("--train-data", type=str, default="data/tokenized/tinystories_train.npy")
    parser.add_argument("--val-data", type=str, default="data/tokenized/tinystories_valid.npy")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda:0 / mps")
    parser.add_argument("--compile", action="store_true", help="CPU 下可用 torch.compile(model) 加速（PDF p.41）")
    parser.add_argument("--seed", type=int, default=42)

    # 模型架构参数（默认对齐 PDF 7.2.1 TinyStories 17M 配置）
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 优化器与学习率调度参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="梯度累积步数（等效 batch_size = batch_size * grad_accum_steps）",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=5000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # 日志与断点保存参数
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints/tinystories_run")
    parser.add_argument("--resume-from", type=str, default=None, help="已有 checkpoint 路径，传入即可断点续训")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps must be >= 1, got {args.grad_accum_steps}")
    torch.manual_seed(args.seed)

    # 1. 创建实验日志与权重保存目录
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"

    # 2. 按 memmap 只读方式挂载训练与验证 token 数据集（几乎不占用额外内存）
    train_data = load_tokens_memmap(ROOT / args.train_data, args.vocab_size)
    val_data = load_tokens_memmap(ROOT / args.val_data, args.vocab_size)
    eff_bs = args.batch_size * args.grad_accum_steps
    print(
        f"Loaded train={len(train_data):,} tokens, val={len(val_data):,} tokens on device={args.device} "
        f"(micro_bs={args.batch_size} x grad_accum={args.grad_accum_steps} -> effective_bs={eff_bs})"
    )

    # 3. 实例化原始模型 raw_model 与 AdamW 优化器
    # 注意保留 raw_model 引用：即使下面开了 torch.compile，保存/恢复权重仍针对原模型，避免键名带 _orig_mod 前缀
    raw_model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )
    optimizer = AdamW(
        raw_model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 4. 如果指定了续训文件，先恢复模型、优化器动量以及起始步数
    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, raw_model, optimizer)
        print(f"Resumed from {args.resume_from} at iteration {start_iter}")

    # 5. 可选 JIT 编译（在 CPU/MPS 下遵循 PDF p.41 建议，可明显加快单步前反向速度）
    if args.compile:
        backend = "aot_eager" if args.device.startswith("mps") else "inductor"
        model: torch.nn.Module = torch.compile(raw_model, backend=backend)
    else:
        model = raw_model

    model.train()
    t0 = time.perf_counter()

    # 6. 核心训练主循环：调度 lr -> 定期评估 -> 采样 -> 前向算 loss -> 清梯度 -> 反向 -> 裁剪 -> 更新
    for it in range(start_iter, args.max_iters):
        # (a) 计算本步余弦退火学习率，并同步写入 optimizer 的每一个 param_group
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # (b) 定期评估训练/验证 loss 与困惑度 (PPL)，写入 JSONL 日志文件方便后续画学习曲线
        if it % args.eval_interval == 0:
            log_eval_metrics(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                eval_iters=args.eval_iters,
                metrics_path=metrics_path,
                step=it,
                t0=t0,
                lr=lr,
                tag="eval",
            )

        # (c) 定期保存断点 checkpoint
        if args.save_interval > 0 and it > start_iter and it % args.save_interval == 0:
            ckpt_file = ckpt_dir / f"ckpt_step_{it}.pt"
            save_checkpoint(raw_model, optimizer, iteration=it, out=ckpt_file)
            print(f"[ckpt] saved {ckpt_file}")

        # (d) 梯度累积：在微批次循环【外】一次性采样完整 effective_bs，再切片喂给微批次！
        # 这样做两个好处：
        #   1. (eff_bs, S) 整数 token ID 仅约百 KB，几乎零额外显存，但能严格保持与“直接大 batch”相同的 RNG 采样序列；
        #   2. 减少 memmap 切片与 H2D 搬运次数，set_to_none=True 也能少一次零填充写内存。
        x_full, y_full = get_batch(train_data, eff_bs, args.context_length, args.device)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro_idx in range(args.grad_accum_steps):
            sl = slice(micro_idx * args.batch_size, (micro_idx + 1) * args.batch_size)
            logits = model(x_full[sl])  # (batch_size, context_length, vocab_size)
            loss = cross_entropy(logits, y_full[sl])
            scaled_loss = loss / args.grad_accum_steps  # 均分后再反传，累加后严格等价大 batch 均值梯度
            accum_loss += loss.item() / args.grad_accum_steps
            scaled_loss.backward()

        # (e) 全部微批次累加完毕后，统一做一次全局 L2 梯度裁剪 -> AdamW 参数更新
        if args.grad_clip > 0.0:
            clip_gradients(raw_model.parameters(), max_l2_norm=args.grad_clip)
        optimizer.step()

        # (f) 定期打印本有效大步的平均训练 loss
        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"[train] step={it:5d} | time={elapsed:7.1f}s | lr={lr:.3e} | loss={accum_loss:.4f}")

    # 7. 训练全部完成后做一次终局验证评估（同步写入 metrics.jsonl）并保存最终模型断点
    final_lr = get_lr_cosine_schedule(
        it=args.max_iters,
        max_learning_rate=args.max_lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
    )
    log_eval_metrics(
        model=model,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        eval_iters=args.eval_iters,
        metrics_path=metrics_path,
        step=args.max_iters,
        t0=t0,
        lr=final_lr,
        tag="done",
    )
    final_ckpt = ckpt_dir / "ckpt_final.pt"
    save_checkpoint(raw_model, optimizer, iteration=args.max_iters, out=final_ckpt)
    print(f"[ckpt] saved final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()