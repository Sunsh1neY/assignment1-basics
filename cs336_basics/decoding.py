from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

# 确保从任意工作目录运行本脚本时都能正确导入根目录下的 cs336_basics
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.architecture import TransformerLM, softmax  # noqa: E402
from cs336_basics.pretokenization import Tokenizer  # noqa: E402


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    """Apply temperature scaling and nucleus (top-p) sampling to next-token logits."""
    # 1. 参数合法性校验：温度必须 >= 0，核采样累计概率阈值 top_p 必须在 [0, 1] 之间
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if not 0.0 <= top_p <= 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")

    # 2. 贪心解码特例：当 temperature == 0（或 top_p == 0）时，直接确定性地取分数最高的 token
    # 这样既严格等价于 τ -> 0 的极限，也彻底避免了除以 0 产生 inf/nan
    if temperature == 0.0 or top_p == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 3. 温度缩放 (Temperature Scaling): softmax(v / τ)（对应 PDF 公式 23）
    # τ < 1 会把分布拉尖（更保守流畅）；τ > 1 会把分布拉平（更有创意但更易胡言乱语）
    # 直接调用我们 Section 3 手写并过测的数值稳定 softmax（内部会先减去最大值）
    probs = softmax(logits / temperature, dim=-1)

    # 4. 如果启用了 Top-p（Nucleus）核采样（top_p < 1.0，对应 PDF 公式 24）：
    # 目标：保留累计概率刚刚达到或超过 top_p 的“最小高概率词集合 V(p)”，截断长尾低概率词
    if top_p < 1.0:
        # (a) 将词表概率按从大到小排序：sorted_probs 与原始下标 sorted_indices 形状均为 (..., vocab_size)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

        # (b) 计算排序后的前缀累计概率和：cum_probs[k] = sum_{j <= k} sorted_probs[j]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # (c) ⚠️ 高频易错点！计算“在加入第 k 个词之前”已经累积了多少概率：
        # 如果在前 k-1 个词时累计和就已经 >= top_p，那么第 k 个词就不需要保留了；
        # 相反，刚刚跨过 top_p 阈值的那个临界词，它前面的和仍 < top_p，所以它会被正确保留下来！
        cumulative_before = cumulative_probs - sorted_probs
        sorted_indices_to_remove = cumulative_before >= top_p

        # (d) 绝对保底：第 0 位（概率最大的头号候选词）无论如何都绝不能被删掉（防御浮点误差）
        sorted_indices_to_remove[..., 0] = False

        # (e) 把被截断的长尾词概率清零，并对保留下来的核心词集合 V(p) 重新归一化使和为 1
        filtered_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        # (f) 在截断后的分布中按概率抽样 1 个位置（得到的是排序后的下标），再 gather 映射回原词表 token ID
        sampled_sorted_idx = torch.multinomial(filtered_probs, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted_idx)

    # 5. 若 top_p == 1.0（不做截断），直接按完整温度缩放后的概率分布多项式抽样 1 个 token ID
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
) -> tuple[list[int], bool]:
    """Autoregressively generate token IDs from a prompt until max_new_tokens or EOS."""
    # 1. 空输入检查：TransformerLM 至少需要 1 个起始 token 才能预测下一个 token
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token ID")

    # 2. 记录并暂存模型原本的训练/评估状态，推理时切换为 eval()，结束时在 finally 中无损恢复
    was_training = model.training
    model.eval()
    stopped_by_eos = False

    try:
        # 3. 探测模型权重当前所在设备（cpu 或 cuda:0），构造形状为 (1, prompt_len) 的输入张量
        device = next(model.parameters()).device
        generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # 4. 自回归循环：每次前向预测 1 个新 token 并拼到序列末尾
        for _ in range(max_new_tokens):
            # 滑动窗口截断：若当前已生成总长度超过模型的最大 context_length（如 256），
            # 只保留最后 context_length 个 token 送入前向传播，防止 RoPE 缓存越界报错
            cond_ids = generated[:, -model.context_length :]

            # 前向计算所有位置的 logits，形状 (1, cur_len, vocab_size)
            logits = model(cond_ids)

            # 只取最后一个时间步（第 cur_len - 1 个位置）对“下一个词”的打分，形状 (1, vocab_size)
            next_token_logits = logits[:, -1, :]

            # 按指定 temperature 与 top_p 采样下一个 token ID，形状 (1, 1)
            next_token = sample_next_token(next_token_logits, temperature=temperature, top_p=top_p)
            next_id = int(next_token.item())

            # 命中结束符 <|endoftext|> 则立即停止生成（不把特殊结束标记本身拼进正文）
            if eos_token_id is not None and next_id == eos_token_id:
                stopped_by_eos = True
                break

            # 将新生成的 token 拼接到序列尾部，作为下一步自回归输入
            generated = torch.cat([generated, next_token], dim=1)

        # 5. 返回完整序列（含 prompt + 新生成内容，或按需切片）与是否自然遇到 EOS 的标志
        return generated[0].tolist(), stopped_by_eos
    finally:
        model.train(was_training)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM (CS336 Section 6)")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/tinystories_run/ckpt_final.pt")
    parser.add_argument("--vocab", type=str, default="artifacts/tinystories_bpe/vocab.json")
    parser.add_argument("--merges", type=str, default="artifacts/tinystories_bpe/merges.json")
    parser.add_argument("--prompt", type=str, default="Once upon a time, there was a little")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # 模型架构超参（默认对齐 TinyStories Run 1 的配置）
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    # 1. 加载我们在 Section 2 训练并导出的 Byte-level BPE 分词器，提取 <|endoftext|> 结束符 ID
    tokenizer = Tokenizer.from_files(ROOT / args.vocab, ROOT / args.merges, special_tokens=["<|endoftext|>"])
    eos_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")

    # 2. 构造与训练时结构完全一致的 TransformerLM，并将参数搬到推理设备
    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )

    # 3. 只加载模型权重（推理无需构造 AdamW 优化器状态，省去多余显存/时间）
    ckpt = torch.load(ROOT / args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    trained_step = ckpt.get("iteration", "?") if isinstance(ckpt, dict) else "?"
    print(f"Loaded checkpoint ({args.checkpoint}, step={trained_step}) on {device} | eos_id={eos_id}")

    # 4. 将用户输入的 prompt 文本编码成整数 ID 序列
    prompt_ids = tokenizer.encode(args.prompt)

    # 5. 自回归采样生成新 token，并用分词器解码回可读英文文本
    all_ids, stopped_by_eos = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )
    new_token_count = len(all_ids) - len(prompt_ids)
    text = tokenizer.decode(all_ids)

    print(
        f"\n--- Generated ({new_token_count} new tokens, stopped_by_eos={stopped_by_eos}, "
        f"T={args.temperature}, top_p={args.top_p}) ---\n"
    )
    print(text)


if __name__ == "__main__":
    main()
