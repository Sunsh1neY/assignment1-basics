import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.pretokenization import Tokenizer, train_bpe  # noqa: E402


def monitor_memory(
    stop_event: threading.Event,
    stats: dict[str, float | int],
    start_time: float,
    interval: float = 0.2,
    log_interval: float = 30.0,
) -> None:
    """Poll parent + worker RSS every `interval` seconds and log every `log_interval` seconds."""
    proc = psutil.Process(os.getpid())
    last_log = start_time
    prev_children = 0
    while not stop_event.is_set():
        try:
            main_rss = proc.memory_info().rss
            children = proc.children(recursive=True)
            child_rss = 0
            max_single_child = 0
            for child in children:
                try:
                    rss = child.memory_info().rss
                    child_rss += rss
                    if rss > max_single_child:
                        max_single_child = rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            total_rss = main_rss + child_rss
            vm = psutil.virtual_memory()
            stats["peak_total_rss_bytes"] = max(stats["peak_total_rss_bytes"], total_rss)
            stats["peak_main_rss_bytes"] = max(stats["peak_main_rss_bytes"], main_rss)
            stats["peak_child_rss_bytes"] = max(stats["peak_child_rss_bytes"], child_rss)
            stats["peak_single_child_rss_bytes"] = max(stats["peak_single_child_rss_bytes"], max_single_child)
            stats["max_children_seen"] = max(stats["max_children_seen"], len(children))
            stats["min_avail_sys_bytes"] = min(stats["min_avail_sys_bytes"], vm.available)

            now = time.perf_counter()
            phase_changed = (prev_children > 0 and len(children) == 0) or (prev_children == 0 and len(children) > 0)
            if phase_changed or (now - last_log >= log_interval):
                note = ""
                if prev_children > 0 and len(children) == 0:
                    note = " [workers exited -> merge phase]"
                elif prev_children == 0 and len(children) > 0:
                    note = " [workers spawned -> pretokenization]"
                print(
                    f"[Monitor t={now - start_time:6.1f}s] workers={len(children)} | "
                    f"main={main_rss / 1e9:.3f}GB | children={child_rss / 1e9:.3f}GB | "
                    f"total={total_rss / 1e9:.3f}GB | peak={stats['peak_total_rss_bytes'] / 1e9:.3f}GB | "
                    f"sys_avail={vm.available / 1e9:.2f}GB{note}",
                    flush=True,
                )
                last_log = now
            prev_children = len(children)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        stop_event.wait(interval)


def _vocab_stats(vocab: dict[int, bytes]) -> dict[str, object]:
    lengths = sorted(len(tok) for tok in vocab.values())
    learned = [tok for tok in vocab.values() if len(tok) > 1 and tok != b"<|endoftext|>"]
    non_ascii = 0
    digit_tokens = 0
    punct_or_symbol_only = 0
    whitespace_or_repeated = 0
    for tok in learned:
        decoded = tok.decode("utf-8", errors="ignore")
        if any(b >= 128 for b in tok):
            non_ascii += 1
        if any(ch.isdigit() for ch in decoded):
            digit_tokens += 1
        if decoded and all(not ch.isalnum() and not ch.isspace() for ch in decoded):
            punct_or_symbol_only += 1
        if decoded and (decoded.isspace() or (len(decoded) >= 4 and len(set(decoded)) <= 2)):
            whitespace_or_repeated += 1
    return {
        "vocab_size": len(vocab),
        "mean_byte_len": round(statistics.fmean(lengths), 4),
        "median_byte_len": statistics.median(lengths),
        "p90_byte_len": lengths[int(0.90 * (len(lengths) - 1))],
        "p99_byte_len": lengths[int(0.99 * (len(lengths) - 1))],
        "max_byte_len": lengths[-1],
        "learned_token_count": len(learned),
        "non_ascii_learned_tokens": non_ascii,
        "digit_containing_learned_tokens": digit_tokens,
        "punct_only_learned_tokens": punct_or_symbol_only,
        "whitespace_or_low_entropy_learned_tokens": whitespace_or_repeated,
    }


def _compression_stats(tokenizer: Tokenizer, sample_path: Path, max_bytes: int = 5_000_000) -> dict[str, float | int]:
    with sample_path.open("rb") as f:
        raw = f.read(max_bytes)
    text = raw.decode("utf-8", errors="ignore")
    byte_len = len(text.encode("utf-8"))
    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    dt = time.perf_counter() - t0
    return {
        "sample_file": sample_path.name,
        "utf8_bytes": byte_len,
        "token_count": len(ids),
        "bytes_per_token": round(byte_len / len(ids), 4) if ids else 0.0,
        "encode_seconds": round(dt, 3),
    }


def compare_with_tinystories(
    owt_tokenizer: Tokenizer,
    ts_dir: Path,
    special_tokens: list[str],
) -> dict[str, object]:
    ts_vocab_path = ts_dir / "vocab.json"
    ts_merges_path = ts_dir / "merges.json"
    if not ts_vocab_path.exists() or not ts_merges_path.exists():
        return {"available": False}

    ts_tok = Tokenizer.from_files(ts_vocab_path, ts_merges_path, special_tokens=special_tokens)
    ts_bytes = set(ts_tok.vocab.values())
    owt_bytes = set(owt_tokenizer.vocab.values())
    shared_all = ts_bytes & owt_bytes
    ts_learned = {tok for tok in ts_bytes if len(tok) > 1 and tok != b"<|endoftext|>"}
    owt_learned = {tok for tok in owt_bytes if len(tok) > 1 and tok != b"<|endoftext|>"}
    shared_learned = ts_learned & owt_learned

    ts_merges_set = set(ts_tok.merges)
    owt_merges_set = set(owt_tokenizer.merges)
    first_n = min(len(ts_tok.merges), len(owt_tokenizer.merges))
    exact_prefix_matches = 0
    for a, b in zip(ts_tok.merges[:first_n], owt_tokenizer.merges[:first_n]):
        if a == b:
            exact_prefix_matches += 1
        else:
            break

    ts_only_longest = sorted(ts_learned - owt_learned, key=lambda b: (len(b), b), reverse=True)[:10]
    owt_only_longest = sorted(owt_learned - ts_learned, key=lambda b: (len(b), b), reverse=True)[:10]

    compression = {}
    ts_valid = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    owt_valid = ROOT / "data" / "owt_valid.txt"
    if ts_valid.exists():
        compression["tinystories_valid_5MB_with_ts_tok"] = _compression_stats(ts_tok, ts_valid)
        compression["tinystories_valid_5MB_with_owt_tok"] = _compression_stats(owt_tokenizer, ts_valid)
    if owt_valid.exists():
        compression["owt_valid_5MB_with_ts_tok"] = _compression_stats(ts_tok, owt_valid)
        compression["owt_valid_5MB_with_owt_tok"] = _compression_stats(owt_tokenizer, owt_valid)

    return {
        "available": True,
        "tinystories_stats": _vocab_stats(ts_tok.vocab),
        "owt_stats": _vocab_stats(owt_tokenizer.vocab),
        "vocab_overlap": {
            "shared_all_tokens": len(shared_all),
            "shared_learned_tokens": len(shared_learned),
            "ts_learned_covered_by_owt_pct": round(100.0 * len(shared_learned) / max(1, len(ts_learned)), 2),
            "jaccard_learned_pct": round(100.0 * len(shared_learned) / max(1, len(ts_learned | owt_learned)), 2),
        },
        "merge_overlap": {
            "ts_merges": len(ts_tok.merges),
            "owt_merges": len(owt_tokenizer.merges),
            "shared_merge_rules": len(ts_merges_set & owt_merges_set),
            "identical_initial_merge_prefix": exact_prefix_matches,
        },
        "sample_ts_only_longest": [repr(b) for b in ts_only_longest],
        "sample_owt_only_longest": [repr(b) for b in owt_only_longest],
        "compression_comparison": compression,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = ROOT / "data" / "owt_train.txt"
    out_dir = ROOT / "artifacts" / "owt_bpe"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.json"
    report_path = out_dir / "report.json"

    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    vm_before = psutil.virtual_memory()
    avail_gb = vm_before.available / 1e9
    num_procs = 4 if avail_gb < 7.5 else 6
    print(
        f"[OWT] Starting train_bpe: input={input_path} ({input_path.stat().st_size / 1e9:.3f} GB), "
        f"vocab_size={vocab_size}, total_ram={vm_before.total / 1e9:.2f} GB, "
        f"avail_ram={avail_gb:.2f} GB, chosen_num_processes={num_procs}",
        flush=True,
    )

    mem_stats: dict[str, float | int] = {
        "peak_total_rss_bytes": 0,
        "peak_main_rss_bytes": 0,
        "peak_child_rss_bytes": 0,
        "peak_single_child_rss_bytes": 0,
        "max_children_seen": 0,
        "min_avail_sys_bytes": vm_before.available,
    }
    t0 = time.perf_counter()
    stop_event = threading.Event()
    mon = threading.Thread(
        target=monitor_memory,
        args=(stop_event, mem_stats, t0, 0.2, 30.0),
        daemon=True,
    )
    mon.start()

    try:
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=num_procs)
    finally:
        t1 = time.perf_counter()
        stop_event.set()
        mon.join(timeout=2.0)

    elapsed = t1 - t0
    print(
        f"[OWT] Training complete in {elapsed:.2f}s ({elapsed / 60:.2f} min). "
        f"Peak total RSS: {mem_stats['peak_total_rss_bytes'] / 1e9:.3f} GB "
        f"(main={mem_stats['peak_main_rss_bytes'] / 1e9:.3f} GB, "
        f"children={mem_stats['peak_child_rss_bytes'] / 1e9:.3f} GB, "
        f"min_sys_avail={mem_stats['min_avail_sys_bytes'] / 1e9:.2f} GB)",
        flush=True,
    )

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(vocab_path, merges_path)
    restored = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    sample = "Hello, world! OpenWebText includes code, URLs https://example.com, numbers 2025, and 中文🙂.<|endoftext|>"
    assert restored.encode(sample) == tokenizer.encode(sample)
    assert restored.decode(restored.encode(sample)) == sample
    assert b"<|endoftext|>" in vocab.values()

    sorted_by_len = sorted(vocab.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    top_longest = [
        {
            "id": tid,
            "len_bytes": len(tok),
            "repr": repr(tok),
            "utf8": tok.decode("utf-8", errors="replace"),
        }
        for tid, tok in sorted_by_len[:20]
    ]
    print(f"[OWT] Saved vocab ({len(vocab)}) & merges ({len(merges)}) to {out_dir}", flush=True)
    print(f"[OWT] Longest token: {top_longest[0]}", flush=True)

    print("[OWT] Comparing OWT tokenizer vs TinyStories tokenizer...", flush=True)
    comparison = compare_with_tinystories(tokenizer, ROOT / "artifacts" / "tinystories_bpe", special_tokens)

    report = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "num_processes": num_procs,
        "avail_ram_before_gb": round(avail_gb, 3),
        "min_avail_sys_ram_gb": round(mem_stats["min_avail_sys_bytes"] / 1e9, 3),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_minutes": round(elapsed / 60.0, 3),
        "peak_total_rss_gb": round(mem_stats["peak_total_rss_bytes"] / 1e9, 3),
        "peak_main_rss_gb": round(mem_stats["peak_main_rss_bytes"] / 1e9, 3),
        "peak_child_rss_gb": round(mem_stats["peak_child_rss_bytes"] / 1e9, 3),
        "peak_single_child_rss_gb": round(mem_stats["peak_single_child_rss_bytes"] / 1e9, 3),
        "max_children_seen": mem_stats["max_children_seen"],
        "top_20_longest_tokens": top_longest,
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "comparison_vs_tinystories": comparison,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OWT] Report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
