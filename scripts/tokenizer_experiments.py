import argparse
import json
import multiprocessing
import os
import random
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.pretokenization import Tokenizer  # noqa: E402
from cs336_basics.pretokenization_example import find_chunk_boundaries  # noqa: E402

SPECIAL_TOKENS = ["<|endoftext|>"]
_WORKER_TOK: Tokenizer | None = None


def _init_encode_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _WORKER_TOK
    _WORKER_TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


def _encode_chunk_worker(args: tuple[str, int, int]) -> bytes:
    path_str, start, end = args
    assert _WORKER_TOK is not None
    with open(path_str, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)
    text = raw.decode("utf-8", errors="ignore")
    ids = _WORKER_TOK.encode(text)
    return np.asarray(ids, dtype=np.uint16).tobytes()


def _sample_documents(file_path: Path, n_docs: int = 10, seed: int = 42, scan_bytes: int = 30_000_000) -> list[str]:
    with file_path.open("rb") as f:
        raw = f.read(scan_bytes)
    text = raw.decode("utf-8", errors="ignore")
    docs = [doc for doc in text.split("<|endoftext|>") if doc.strip()]
    # Drop trailing truncated document if file exceeded scan_bytes
    if file_path.stat().st_size > scan_bytes and len(docs) > n_docs:
        docs = docs[:-1]
    rng = random.Random(seed)
    return rng.sample(docs, n_docs)


def _encode_docs_stats(tok: Tokenizer, docs: list[str]) -> dict[str, object]:
    doc_bytes = []
    doc_tokens = []
    doc_ratios = []
    sample_preview = None
    for idx, doc in enumerate(docs):
        b_len = len(doc.encode("utf-8"))
        ids = tok.encode(doc)
        assert tok.decode(ids) == doc
        t_len = len(ids)
        doc_bytes.append(b_len)
        doc_tokens.append(t_len)
        doc_ratios.append(b_len / t_len)
        if idx == 0:
            snippet = doc.strip().replace("\r\n", "\n")[:220]
            snippet_ids = tok.encode(snippet)
            sample_preview = {
                "snippet": snippet,
                "token_count": len(snippet_ids),
                "pieces": [tok.vocab[tid].decode("utf-8", errors="replace") for tid in snippet_ids[:35]],
            }
    total_b = sum(doc_bytes)
    total_t = sum(doc_tokens)
    return {
        "num_docs": len(docs),
        "total_utf8_bytes": total_b,
        "total_tokens": total_t,
        "overall_bytes_per_token": round(total_b / total_t, 4),
        "mean_doc_bytes_per_token": round(statistics.fmean(doc_ratios), 4),
        "min_doc_bytes_per_token": round(min(doc_ratios), 4),
        "max_doc_bytes_per_token": round(max(doc_ratios), 4),
        "first_doc_preview": sample_preview,
    }


def measure_throughput(ts_tok: Tokenizer, owt_tok: Tokenizer) -> dict[str, object]:
    pile_bytes_si = 825 * 1_000_000_000
    pile_bytes_iec = 825 * (1024**3)
    ts_valid = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    owt_valid = ROOT / "data" / "owt_valid.txt"

    results: dict[str, object] = {}
    for label, tok, path in [
        ("tinystories_10k", ts_tok, ts_valid),
        ("owt_32k", owt_tok, owt_valid),
    ]:
        with path.open("rb") as f:
            sample_text = f.read(5_000_000).decode("utf-8", errors="ignore")
        byte_len = len(sample_text.encode("utf-8"))
        # Warmup small prefix so dict/regex imports are settled, then measure fresh Tokenizer throughput too
        fresh_tok = Tokenizer(tok.vocab, tok.merges, tok.special_tokens)

        t0 = time.perf_counter()
        ids = fresh_tok.encode(sample_text)
        dt_encode = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = fresh_tok.encode(sample_text)
        dt_warm = time.perf_counter() - t1

        # Also measure encode_iterable line-by-line on first 1 MB
        lines_1mb = sample_text[:1_000_000].splitlines(keepends=True)
        iter_bytes = sum(len(line.encode("utf-8")) for line in lines_1mb)
        stream_tok = Tokenizer(tok.vocab, tok.merges, tok.special_tokens)
        t2 = time.perf_counter()
        stream_count = sum(1 for _ in stream_tok.encode_iterable(lines_1mb))
        dt_stream = time.perf_counter() - t2

        bps = byte_len / dt_encode
        warm_bps = byte_len / dt_warm
        stream_bps = iter_bytes / dt_stream
        results[label] = {
            "sample_utf8_bytes": byte_len,
            "sample_tokens": len(ids),
            "single_proc_cold_seconds": round(dt_encode, 3),
            "single_proc_cold_bytes_per_sec": round(bps, 1),
            "single_proc_cold_mb_per_sec": round(bps / 1e6, 3),
            "single_proc_warm_mb_per_sec": round(warm_bps / 1e6, 3),
            "encode_iterable_1mb_tokens": stream_count,
            "encode_iterable_line_mb_per_sec": round(stream_bps / 1e6, 3),
            "pile_825gb_si_hours_single_proc": round(pile_bytes_si / bps / 3600, 2),
            "pile_825gib_hours_single_proc": round(pile_bytes_iec / bps / 3600, 2),
        }
    return results


def _monitor_rss(stop_event: threading.Event, stats: dict[str, float | int], t0: float, label: str) -> None:
    proc = psutil.Process(os.getpid())
    last_log = t0
    while not stop_event.is_set():
        try:
            main_rss = proc.memory_info().rss
            children = proc.children(recursive=True)
            child_rss = 0
            for c in children:
                try:
                    child_rss += c.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            total = main_rss + child_rss
            stats["peak_total_rss_bytes"] = max(stats["peak_total_rss_bytes"], total)
            now = time.perf_counter()
            if now - last_log >= 30.0:
                vm = psutil.virtual_memory()
                print(
                    f"[{label} t={now - t0:6.1f}s] workers={len(children)} | "
                    f"total={total / 1e9:.3f}GB | peak={stats['peak_total_rss_bytes'] / 1e9:.3f}GB | "
                    f"sys_avail={vm.available / 1e9:.2f}GB",
                    flush=True,
                )
                last_log = now
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        stop_event.wait(0.25)


def encode_file_to_uint16_npy(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    vocab_size: int,
    num_workers: int,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = output_path.with_suffix(".bin.tmp")
    file_size = input_path.stat().st_size
    desired_chunks = max(1, (file_size + chunk_bytes - 1) // chunk_bytes)

    with input_path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, b"<|endoftext|>")
    ranges = [
        (str(input_path), boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]

    workers = min(num_workers, len(ranges))
    t0 = time.perf_counter()
    mem_stats: dict[str, float | int] = {"peak_total_rss_bytes": 0}
    stop_event = threading.Event()
    mon = threading.Thread(
        target=_monitor_rss,
        args=(stop_event, mem_stats, t0, f"Encode:{input_path.name}"),
        daemon=True,
    )
    mon.start()

    total_tokens = 0
    try:
        with tmp_bin.open("wb") as f_bin:
            if workers <= 1:
                _init_encode_worker(str(vocab_path), str(merges_path), SPECIAL_TOKENS)
                for chunk_args in ranges:
                    raw_u16 = _encode_chunk_worker(chunk_args)
                    f_bin.write(raw_u16)
                    total_tokens += len(raw_u16) // 2
            else:
                with multiprocessing.Pool(
                    processes=workers,
                    initializer=_init_encode_worker,
                    initargs=(str(vocab_path), str(merges_path), SPECIAL_TOKENS),
                ) as pool:
                    for raw_u16 in pool.imap(_encode_chunk_worker, ranges, chunksize=1):
                        f_bin.write(raw_u16)
                        total_tokens += len(raw_u16) // 2

        with output_path.open("wb") as f_out, tmp_bin.open("rb") as f_in:
            np.lib.format.write_array_header_1_0(
                f_out,
                {"descr": np.dtype("<u2").str, "fortran_order": False, "shape": (total_tokens,)},
            )
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    finally:
        stop_event.set()
        mon.join(timeout=2.0)
        if tmp_bin.exists():
            tmp_bin.unlink()

    elapsed = time.perf_counter() - t0

    # Verify memory-mapped readback and token bounds
    arr = np.load(output_path, mmap_mode="r")
    assert arr.dtype == np.uint16
    assert arr.shape == (total_tokens,)
    max_id = int(arr.max()) if total_tokens > 0 else -1
    assert max_id < vocab_size, f"Token ID {max_id} >= vocab_size {vocab_size}"
    del arr

    out_bytes = output_path.stat().st_size
    throughput_mb_s = file_size / 1e6 / elapsed
    print(
        f"[Encode] {input_path.name} -> {output_path.name}: "
        f"{total_tokens:,} tokens ({out_bytes / 1e6:.2f} MB) in {elapsed:.2f}s "
        f"({throughput_mb_s:.2f} MB/s, workers={workers}, max_id={max_id}, "
        f"peak_rss={mem_stats['peak_total_rss_bytes'] / 1e9:.3f} GB)",
        flush=True,
    )
    return {
        "input_file": str(input_path),
        "input_bytes": file_size,
        "output_file": str(output_path),
        "output_npy_bytes": out_bytes,
        "num_chunks": len(ranges),
        "workers": workers,
        "total_tokens": total_tokens,
        "bytes_per_token": round(file_size / total_tokens, 4) if total_tokens else 0.0,
        "max_token_id": max_id,
        "elapsed_seconds": round(elapsed, 3),
        "parallel_mb_per_sec": round(throughput_mb_s, 3),
        "peak_rss_gb": round(mem_stats["peak_total_rss_bytes"] / 1e9, 3),
    }


def run_eval_and_valid(ts_tok: Tokenizer, owt_tok: Tokenizer, out_dir: Path, num_workers: int) -> dict[str, object]:
    ts_valid = ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt"
    owt_valid = ROOT / "data" / "owt_valid.txt"
    ts_v = ROOT / "artifacts" / "tinystories_bpe" / "vocab.json"
    ts_m = ROOT / "artifacts" / "tinystories_bpe" / "merges.json"
    owt_v = ROOT / "artifacts" / "owt_bpe" / "vocab.json"
    owt_m = ROOT / "artifacts" / "owt_bpe" / "merges.json"

    print("[2.7 a/b] Sampling 10 documents from TinyStories and OpenWebText...", flush=True)
    ts_docs = _sample_documents(ts_valid, n_docs=10, seed=42)
    owt_docs = _sample_documents(owt_valid, n_docs=10, seed=42)

    part_a_ts = _encode_docs_stats(ts_tok, ts_docs)
    part_a_owt = _encode_docs_stats(owt_tok, owt_docs)
    part_b_cross = _encode_docs_stats(ts_tok, owt_docs)
    part_b_inflation = round(
        100.0 * (part_b_cross["total_tokens"] - part_a_owt["total_tokens"]) / part_a_owt["total_tokens"],
        2,
    )

    print("[2.7 c] Measuring single-process and streaming throughput...", flush=True)
    part_c = measure_throughput(ts_tok, owt_tok)

    print("[2.7 d] Encoding validation datasets and verifying chunk equivalence...", flush=True)
    ts_valid_npy = out_dir / "tinystories_valid.npy"
    owt_valid_npy = out_dir / "owt_valid.npy"
    ts_valid_stat = encode_file_to_uint16_npy(ts_valid, ts_valid_npy, ts_v, ts_m, 10000, num_workers)
    owt_valid_stat = encode_file_to_uint16_npy(owt_valid, owt_valid_npy, owt_v, owt_m, 32000, num_workers)

    # Bit-for-bit verification on TinyStoriesV2-GPT4-valid.txt (22.5 MB)
    serial_ids = np.asarray(ts_tok.encode(ts_valid.read_text(encoding="utf-8")), dtype=np.uint16)
    parallel_ids = np.load(ts_valid_npy, mmap_mode="r")
    assert np.array_equal(serial_ids, parallel_ids), "Parallel chunk encoding differed from serial encode!"
    print(f"[Verify] Bit-for-bit match verified on {ts_valid.name} ({len(serial_ids):,} tokens).", flush=True)

    return {
        "part_a_10docs_in_domain": {
            "tinystories_tok_on_tinystories": part_a_ts,
            "owt_tok_on_owt": part_a_owt,
        },
        "part_b_ts_tok_on_owt_10docs": {
            "stats": part_b_cross,
            "token_count_increase_pct_vs_owt_tok": part_b_inflation,
            "owt_tok_preview_same_first_doc": part_a_owt["first_doc_preview"],
        },
        "part_c_throughput_and_pile_estimate": part_c,
        "part_d_encoded_datasets": {
            "tinystories_valid": ts_valid_stat,
            "owt_valid": owt_valid_stat,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run Section 2.7 tokenizer experiments and dataset encoding.")
    parser.add_argument("--mode", choices=["eval", "encode-train", "all"], default="all")
    args = parser.parse_args()

    ts_v = ROOT / "artifacts" / "tinystories_bpe" / "vocab.json"
    ts_m = ROOT / "artifacts" / "tinystories_bpe" / "merges.json"
    owt_v = ROOT / "artifacts" / "owt_bpe" / "vocab.json"
    owt_m = ROOT / "artifacts" / "owt_bpe" / "merges.json"

    report_dir = ROOT / "artifacts" / "tokenizer_experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    tokenized_dir = ROOT / "data" / "tokenized"

    avail_gb = psutil.virtual_memory().available / 1e9
    num_workers = min(6, os.cpu_count() or 1) if avail_gb >= 4.5 else min(4, os.cpu_count() or 1)
    print(f"[2.7] mode={args.mode}, avail_ram={avail_gb:.2f}GB, workers={num_workers}", flush=True)

    report: dict[str, object] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    if args.mode in ("eval", "all"):
        ts_tok = Tokenizer.from_files(ts_v, ts_m, special_tokens=SPECIAL_TOKENS)
        owt_tok = Tokenizer.from_files(owt_v, owt_m, special_tokens=SPECIAL_TOKENS)
        report = run_eval_and_valid(ts_tok, owt_tok, tokenized_dir, num_workers)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[2.7] Saved eval + valid report to {report_path}", flush=True)

    if args.mode in ("encode-train", "all"):
        datasets = report.setdefault("part_d_encoded_datasets", {})
        assert isinstance(datasets, dict)
        datasets["tinystories_train"] = encode_file_to_uint16_npy(
            ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
            tokenized_dir / "tinystories_train.npy",
            ts_v,
            ts_m,
            10000,
            num_workers,
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        datasets["owt_train"] = encode_file_to_uint16_npy(
            ROOT / "data" / "owt_train.txt",
            tokenized_dir / "owt_train.npy",
            owt_v,
            owt_m,
            32000,
            num_workers,
        )
        # Update combined multi-proc Pile estimate using observed OWT train parallel throughput
        owt_train_stat = datasets["owt_train"]
        pile_bytes_si = 825 * 1_000_000_000
        bps_parallel = owt_train_stat["input_bytes"] / owt_train_stat["elapsed_seconds"]
        owt_train_stat["pile_825gb_si_hours_parallel"] = round(pile_bytes_si / bps_parallel / 3600, 2)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[2.7] Complete report updated at {report_path}", flush=True)


if __name__ == "__main__":
    main()
