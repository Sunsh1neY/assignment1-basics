import cProfile
import io
import json
import os
import pstats
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.pretokenization import Tokenizer, count_chunk_tokens, train_bpe


def monitor_memory(stop_event: threading.Event, stats: dict[str, float], interval: float = 0.2) -> None:
    proc = psutil.Process(os.getpid())
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
            stats["peak_total_rss_bytes"] = max(stats["peak_total_rss_bytes"], total_rss)
            stats["peak_main_rss_bytes"] = max(stats["peak_main_rss_bytes"], main_rss)
            stats["peak_child_rss_bytes"] = max(stats["peak_child_rss_bytes"], child_rss)
            stats["peak_single_child_rss_bytes"] = max(stats["peak_single_child_rss_bytes"], max_single_child)
            stats["max_children_seen"] = max(stats["max_children_seen"], len(children))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        stop_event.wait(interval)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    input_path = ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
    out_dir = ROOT / "artifacts" / "tinystories_bpe"
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.json"
    report_path = out_dir / "report.json"
    profile_path = out_dir / "cprofile_stats.txt"

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vm_before = psutil.virtual_memory()
    print(
        f"[Phase 1] Starting TinyStories train_bpe: input={input_path} "
        f"({input_path.stat().st_size / 1e9:.3f} GB), vocab_size={vocab_size}, "
        f"avail_ram={vm_before.available / 1e9:.2f} GB",
        flush=True,
    )

    mem_stats = {
        "peak_total_rss_bytes": 0,
        "peak_main_rss_bytes": 0,
        "peak_child_rss_bytes": 0,
        "peak_single_child_rss_bytes": 0,
        "max_children_seen": 0,
    }
    stop_event = threading.Event()
    mon = threading.Thread(target=monitor_memory, args=(stop_event, mem_stats, 0.2), daemon=True)
    mon.start()

    t0 = time.perf_counter()
    try:
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    finally:
        t1 = time.perf_counter()
        stop_event.set()
        mon.join(timeout=2.0)

    elapsed_clean = t1 - t0
    print(
        f"[Phase 1] Done in {elapsed_clean:.2f}s ({elapsed_clean / 60:.2f} min). "
        f"Peak total RSS (main+workers): {mem_stats['peak_total_rss_bytes'] / 1e9:.3f} GB, "
        f"Peak main RSS: {mem_stats['peak_main_rss_bytes'] / 1e9:.3f} GB, "
        f"Max workers: {mem_stats['max_children_seen']}",
        flush=True,
    )

    tokenizer = Tokenizer(vocab, merges, special_tokens)
    tokenizer.save(vocab_path, merges_path)
    restored = Tokenizer.from_files(vocab_path, merges_path)
    sample = "Once upon a time, there was a little girl named Lily.<|endoftext|>"
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
        for tid, tok in sorted_by_len[:15]
    ]
    print(f"[Phase 1] Saved vocab ({len(vocab)}) & merges ({len(merges)}) to {out_dir}", flush=True)
    print(f"[Phase 1] Longest token: {top_longest[0]}", flush=True)

    print("[Phase 2] Running cProfile on TinyStories train_bpe...", flush=True)
    pr = cProfile.Profile()
    t2 = time.perf_counter()
    pr.enable()
    train_bpe(input_path, vocab_size, special_tokens)
    pr.disable()
    t3 = time.perf_counter()

    cum_buf = io.StringIO()
    pstats.Stats(pr, stream=cum_buf).sort_stats("cumtime").print_stats(25)
    tot_buf = io.StringIO()
    pstats.Stats(pr, stream=tot_buf).sort_stats("tottime").print_stats(25)

    valid_path = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    pr_worker = cProfile.Profile()
    pr_worker.enable()
    count_chunk_tokens((valid_path, 0, valid_path.stat().st_size, special_tokens))
    pr_worker.disable()
    worker_buf = io.StringIO()
    pstats.Stats(pr_worker, stream=worker_buf).sort_stats("cumtime").print_stats(20)

    profile_text = (
        f"=== Main process train_bpe cProfile (total {t3 - t2:.2f}s) - by cumtime ===\n"
        + cum_buf.getvalue()
        + "\n=== Main process train_bpe cProfile - by tottime ===\n"
        + tot_buf.getvalue()
        + "\n=== Single-worker count_chunk_tokens cProfile (on TinyStories valid 22.5MB) - by cumtime ===\n"
        + worker_buf.getvalue()
    )
    profile_path.write_text(profile_text, encoding="utf-8")

    report = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds_clean": elapsed_clean,
        "elapsed_seconds_profiled": t3 - t2,
        "peak_total_rss_gb": mem_stats["peak_total_rss_bytes"] / 1e9,
        "peak_main_rss_gb": mem_stats["peak_main_rss_bytes"] / 1e9,
        "peak_child_rss_gb": mem_stats["peak_child_rss_bytes"] / 1e9,
        "peak_single_child_rss_gb": mem_stats["peak_single_child_rss_bytes"] / 1e9,
        "max_children_seen": mem_stats["max_children_seen"],
        "top_15_longest_tokens": top_longest,
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
        "profile_path": str(profile_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Phase 2] Profiling & report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
