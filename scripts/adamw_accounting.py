import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_adamw_accounting(
    name: str,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: float | int,
    ce_saves_vocab_logits: bool = False,
) -> dict[str, object]:
    """Compute exact float32 AdamW memory, step FLOPs, and H100 training time."""
    v = vocab_size
    s = context_length
    l_cnt = num_layers
    d = d_model
    h = num_heads

    # 1. Parameters P (elements)
    emb_params = v * d
    lm_head_params = v * d
    attn_params = l_cnt * 4 * d * d
    ffn_params = l_cnt * 3 * d * d_ff
    rmsnorm_params = l_cnt * 2 * d + d
    total_params = emb_params + lm_head_params + attn_params + ffn_params + rmsnorm_params

    # Float32: 4 bytes per element
    # Static memory (b): Params (1P) + Grads (1P) + Optimizer m,v (2P) = 4P elements = 16P bytes
    params_bytes = 4 * total_params
    grads_bytes = 4 * total_params
    opt_state_bytes = 8 * total_params
    static_bytes_b = params_bytes + grads_bytes + opt_state_bytes

    # 2. Activations A (elements per batch item B=1)
    act_block_rmsnorm = 2 * s * d
    act_block_qkv = 3 * s * d
    act_block_qk = h * s * s
    act_block_softmax = h * s * s
    act_block_av = s * d
    act_block_out_proj = s * d
    act_block_swiglu_w1 = s * d_ff
    act_block_swiglu_silu = s * d_ff
    act_block_swiglu_w3 = s * d_ff
    act_block_swiglu_mul = s * d_ff
    act_block_swiglu_w2 = s * d

    act_block_total = (
        act_block_rmsnorm
        + act_block_qkv
        + act_block_qk
        + act_block_softmax
        + act_block_av
        + act_block_out_proj
        + act_block_swiglu_w1
        + act_block_swiglu_silu
        + act_block_swiglu_w3
        + act_block_swiglu_mul
        + act_block_swiglu_w2
    )
    act_all_blocks = l_cnt * act_block_total
    act_final_rmsnorm = s * d
    act_lm_head = s * v
    act_cross_entropy = s * v if ce_saves_vocab_logits else s

    act_elems_per_batch = act_all_blocks + act_final_rmsnorm + act_lm_head + act_cross_entropy
    act_bytes_per_batch_a = 4 * act_elems_per_batch

    limit_80gb_si = 80_000_000_000
    limit_80gib_iec = 80 * (1024**3)
    max_bs_si = int((limit_80gb_si - static_bytes_b) // act_bytes_per_batch_a)
    max_bs_iec = int((limit_80gib_iec - static_bytes_b) // act_bytes_per_batch_a)

    # 3. FLOPs accounting
    # AdamW FLOPs per parameter element per step:
    #   weight decay (theta - lr*wd*theta): 2, m update: 3, v update: 4, step update: 5 -> 14P
    adamw_step_flops = 14 * total_params

    # Forward matmul FLOPs for B=1, S=context_length (from Section 3.5 rule: 2mnp)
    fwd_qkv_proj = l_cnt * 3 * (2 * s * d * d)
    fwd_qk_av = l_cnt * 2 * (2 * s * s * d)
    fwd_o_proj = l_cnt * (2 * s * d * d)
    fwd_ffn = l_cnt * 3 * (2 * s * d * d_ff)
    fwd_lm_head = 2 * s * d * v
    fwd_flops_per_seq = fwd_qkv_proj + fwd_qk_av + fwd_o_proj + fwd_ffn + fwd_lm_head
    fwd_bwd_flops_per_seq = 3 * fwd_flops_per_seq

    # Part (d): 400K steps, batch_size=1024, H100 TF32 495 TF/s @ 50% MFU
    steps = 400_000
    train_bs = 1024
    h100_peak_tf = 495e12
    mfu = 0.50
    effective_flops_per_sec = h100_peak_tf * mfu

    step_compute_flops = train_bs * fwd_bwd_flops_per_seq
    total_train_compute_flops = steps * step_compute_flops
    total_train_with_opt_flops = steps * (step_compute_flops + adamw_step_flops)

    hours_compute_only = total_train_compute_flops / effective_flops_per_sec / 3600.0
    hours_with_adamw = total_train_with_opt_flops / effective_flops_per_sec / 3600.0

    return {
        "name": name,
        "config": {
            "vocab_size": v,
            "context_length": s,
            "num_layers": l_cnt,
            "d_model": d,
            "num_heads": h,
            "d_ff": d_ff,
            "ce_saves_vocab_logits": ce_saves_vocab_logits,
        },
        "memory_static": {
            "total_params": total_params,
            "params_bytes": params_bytes,
            "grads_bytes": grads_bytes,
            "optimizer_state_bytes": opt_state_bytes,
            "total_static_bytes_b": static_bytes_b,
            "total_static_gb": round(static_bytes_b / 1e9, 4),
            "total_static_gib": round(static_bytes_b / (1024**3), 4),
        },
        "memory_activations_per_batch": {
            "elements_per_batch": act_elems_per_batch,
            "bytes_per_batch_a": act_bytes_per_batch_a,
            "gb_per_batch": round(act_bytes_per_batch_a / 1e9, 4),
            "gib_per_batch": round(act_bytes_per_batch_a / (1024**3), 4),
            "breakdown_bytes_per_batch": {
                "blocks_rmsnorm": 4 * l_cnt * act_block_rmsnorm,
                "blocks_mha_qkv_proj": 4 * l_cnt * act_block_qkv,
                "blocks_mha_qk_matmul": 4 * l_cnt * act_block_qk,
                "blocks_mha_softmax": 4 * l_cnt * act_block_softmax,
                "blocks_mha_av_and_oproj": 4 * l_cnt * (act_block_av + act_block_out_proj),
                "blocks_swiglu_all": 4 * l_cnt * (4 * s * d_ff + s * d),
                "final_rmsnorm": 4 * act_final_rmsnorm,
                "output_embedding_lm_head": 4 * act_lm_head,
                "cross_entropy": 4 * act_cross_entropy,
            },
        },
        "peak_memory_80gb": {
            "formula_bytes": f"{act_bytes_per_batch_a:.0f} * B + {static_bytes_b:.0f}",
            "formula_gb_si": f"{act_bytes_per_batch_a / 1e9:.4f} * B + {static_bytes_b / 1e9:.4f} GB",
            "formula_gib_iec": f"{act_bytes_per_batch_a / (1024**3):.4f} * B + {static_bytes_b / (1024**3):.4f} GiB",
            "max_batch_size_80gb_si": max_bs_si,
            "max_batch_size_80gib_iec": max_bs_iec,
            "peak_at_max_bs_gb_si": round((act_bytes_per_batch_a * max_bs_si + static_bytes_b) / 1e9, 3),
            "peak_at_max_bs_gib_iec": round((act_bytes_per_batch_a * max_bs_iec + static_bytes_b) / (1024**3), 3),
        },
        "flops_and_training_time": {
            "adamw_step_flops_14p": adamw_step_flops,
            "fwd_flops_per_seq": fwd_flops_per_seq,
            "fwd_bwd_flops_per_seq": fwd_bwd_flops_per_seq,
            "step_fwd_bwd_flops_bs1024": step_compute_flops,
            "adamw_share_of_bs1024_step_pct": adamw_step_flops / step_compute_flops * 100.0,
            "total_tokens_400k_bs1024": steps * train_bs * s,
            "total_train_flops_compute_only": total_train_compute_flops,
            "h100_effective_flops_per_sec": effective_flops_per_sec,
            "hours_on_1_h100": round(hours_compute_only, 2),
            "days_on_1_h100": round(hours_compute_only / 24.0, 2),
            "hours_on_1_h100_with_adamw": round(hours_with_adamw, 2),
        },
    }


def main() -> None:
    out_dir = ROOT / "artifacts" / "adamw_accounting"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical PDF spec uses d_ff = 8/3 * d_model, plus actual Section 3.5 d_ff = 4288
    xl_canonical = compute_adamw_accounting(
        name="GPT-2 XL (d_ff = 8/3 * d_model, CE output B*S)",
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=(8 * 1600) / 3,
        ce_saves_vocab_logits=False,
    )
    xl_canonical_ce_vocab = compute_adamw_accounting(
        name="GPT-2 XL (d_ff = 8/3 * d_model, CE saves B*S*V)",
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=(8 * 1600) / 3,
        ce_saves_vocab_logits=True,
    )
    xl_aligned64 = compute_adamw_accounting(
        name="GPT-2 XL (d_ff = 4288, CE output B*S)",
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=4288,
        ce_saves_vocab_logits=False,
    )

    report = {
        "gpt2_xl_canonical": xl_canonical,
        "gpt2_xl_canonical_ce_vocab": xl_canonical_ce_vocab,
        "gpt2_xl_aligned64": xl_aligned64,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(json.dumps(xl_canonical, indent=2))
    print("\n--- Aligned 64 (d_ff=4288) ---")
    print(json.dumps(xl_aligned64["peak_memory_80gb"], indent=2))
    print(json.dumps(xl_aligned64["flops_and_training_time"], indent=2))


if __name__ == "__main__":
    main()
