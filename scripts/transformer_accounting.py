import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.architecture import TransformerLM  # noqa: E402


def _nearest_64(d_model: int) -> int:
    return max(64, int(round((8 * d_model / 3) / 64)) * 64)


def compute_accounting(
    name: str,
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int | None = None,
    verify_with_model: bool = False,
) -> dict[str, object]:
    if d_ff is None:
        d_ff = _nearest_64(d_model)
    s = context_length
    v = vocab_size
    l_cnt = num_layers
    d = d_model

    # Trainable parameters
    emb_params = v * d
    lm_head_params = v * d
    attn_params_per_layer = 4 * d * d
    ffn_params_per_layer = 3 * d * d_ff
    ln_params_per_layer = 2 * d
    final_ln_params = d
    total_params = (
        emb_params
        + lm_head_params
        + l_cnt * (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer)
        + final_ln_params
    )

    if verify_with_model:
        # Instantiate on CPU only when small enough (or check exact counts)
        import torch

        with torch.device("meta"):
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            actual = sum(p.numel() for p in model.parameters())
            assert actual == total_params, f"{name}: formula {total_params} != model {actual}"

    fp32_bytes = 4 * total_params

    # Forward matmul FLOPs (batch_size = 1, sequence_length = context_length)
    qkv_proj_flops_layer = 3 * (2 * s * d * d)
    qk_matmul_flops_layer = 2 * s * s * d
    av_matmul_flops_layer = 2 * s * s * d
    o_proj_flops_layer = 2 * s * d * d
    ffn_flops_layer = 3 * (2 * s * d * d_ff)

    attn_proj_flops = l_cnt * (qkv_proj_flops_layer + o_proj_flops_layer)
    attn_seq_flops = l_cnt * (qk_matmul_flops_layer + av_matmul_flops_layer)
    attn_total_flops = attn_proj_flops + attn_seq_flops
    ffn_total_flops = l_cnt * ffn_flops_layer
    lm_head_flops = 2 * s * d * v
    total_flops = attn_total_flops + ffn_total_flops + lm_head_flops

    def pct(x: int) -> float:
        return round(100.0 * x / total_flops, 2)

    return {
        "name": name,
        "config": {
            "vocab_size": v,
            "context_length": s,
            "num_layers": l_cnt,
            "d_model": d,
            "num_heads": num_heads,
            "d_ff": d_ff,
        },
        "parameters": {
            "total": total_params,
            "embeddings_plus_lm_head": emb_params + lm_head_params,
            "attention_all_layers": l_cnt * attn_params_per_layer,
            "ffn_all_layers": l_cnt * ffn_params_per_layer,
            "rmsnorm_all": l_cnt * ln_params_per_layer + final_ln_params,
            "fp32_bytes": fp32_bytes,
            "fp32_gb": round(fp32_bytes / 1e9, 4),
            "fp32_gib": round(fp32_bytes / (1024**3), 4),
        },
        "flops_per_forward": {
            "total": total_flops,
            "total_tflops": round(total_flops / 1e12, 4),
            "breakdown": {
                "attn_qkv_proj": l_cnt * qkv_proj_flops_layer,
                "attn_qk_matmul": l_cnt * qk_matmul_flops_layer,
                "attn_av_matmul": l_cnt * av_matmul_flops_layer,
                "attn_output_proj": l_cnt * o_proj_flops_layer,
                "attn_proj_total": attn_proj_flops,
                "attn_seq_total": attn_seq_flops,
                "attn_all": attn_total_flops,
                "ffn_w1_w2_w3": ffn_total_flops,
                "lm_head": lm_head_flops,
            },
            "share_pct": {
                "attn_qkv_proj": pct(l_cnt * qkv_proj_flops_layer),
                "attn_qk_av_seq": pct(attn_seq_flops),
                "attn_output_proj": pct(l_cnt * o_proj_flops_layer),
                "attn_proj_total": pct(attn_proj_flops),
                "attn_all": pct(attn_total_flops),
                "ffn_swiglu": pct(ffn_total_flops),
                "lm_head": pct(lm_head_flops),
            },
        },
    }


def main() -> None:
    specs = [
        ("GPT-2 small (1024)", 50257, 1024, 12, 768, 12, None),
        ("GPT-2 medium (1024)", 50257, 1024, 24, 1024, 16, None),
        ("GPT-2 large (1024)", 50257, 1024, 36, 1280, 20, None),
        ("GPT-2 XL (1024)", 50257, 1024, 48, 1600, 25, 4288),
        ("GPT-2 XL (16384)", 50257, 16384, 48, 1600, 25, 4288),
    ]

    results = []
    for name, v, s, l_cnt, d, h, d_ff in specs:
        # Check meta device count (catch if trunc_normal_ dislikes meta)
        res = compute_accounting(name, v, s, l_cnt, d, h, d_ff, verify_with_model=False)
        results.append(res)

    # Verify formula against actual CPU small TransformerLM instance too
    import torch

    small_check = compute_accounting("sanity", 1000, 32, 3, 64, 4, 128, verify_with_model=False)
    m = TransformerLM(vocab_size=1000, context_length=32, d_model=64, num_layers=3, num_heads=4, d_ff=128)
    actual_small = sum(p.numel() for p in m.parameters())
    assert actual_small == small_check["parameters"]["total"], (
        f"Mismatch {actual_small} vs {small_check['parameters']['total']}"
    )
    with torch.no_grad():
        _ = m(torch.randint(0, 1000, (1, 32)))

    out_dir = ROOT / "artifacts" / "transformer_accounting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    payload = {
        "models": results,
        "xl_16384_vs_1024_flops_ratio": round(
            results[4]["flops_per_forward"]["total"] / results[3]["flops_per_forward"]["total"], 3
        ),
        "verified_formula_against_real_model": True,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved accounting report to {out_path}")
    for r in results:
        p = r["parameters"]
        f = r["flops_per_forward"]
        sh = f["share_pct"]
        print(
            f"{r['name']:22s} | d_ff={r['config']['d_ff']:4d} | params={p['total']:13,d} "
            f"({p['fp32_gb']:.3f}GB / {p['fp32_gib']:.3f}GiB) | FLOPs={f['total_tflops']:7.4f} TF | "
            f"AttnProj={sh['attn_proj_total']:5.2f}% AttnSeq={sh['attn_qk_av_seq']:5.2f}% "
            f"FFN={sh['ffn_swiglu']:5.2f}% LMHead={sh['lm_head']:5.2f}%"
        )


if __name__ == "__main__":
    main()
