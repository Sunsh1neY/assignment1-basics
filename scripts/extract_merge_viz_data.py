import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.pretokenization import _PRETOKEN, Tokenizer  # noqa: E402


def _fmt_bytes(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
        # Make whitespace visible cleanly while keeping normal text readable
        out = []
        for ch in s:
            if ch == " ":
                out.append("␠")
            elif ch == "\n":
                out.append("↵")
            elif ch == "\t":
                out.append("⇥")
            elif ord(ch) < 32 or ord(ch) == 127:
                out.append(f"\\x{ord(ch):02x}")
            else:
                out.append(ch)
        return "".join(out)
    except UnicodeDecodeError:
        return "0x" + b.hex()


def _classify_merge(a: bytes, b: bytes) -> str:
    if len(a) == 1 and len(b) == 1:
        return "byte_byte"
    merged = a + b
    if any(x >= 128 for x in merged):
        return "non_ascii"
    s = merged.decode("ascii")
    if s.isspace():
        return "whitespace"
    if any(ch.isdigit() for ch in s):
        return "digit"
    stripped = s[1:] if s.startswith(" ") else s
    if stripped and stripped.isalpha():
        return "ascii_word"
    if all(not ch.isalnum() and not ch.isspace() for ch in stripped):
        return "punct_sym"
    return "mixed"


def _analyze_merges(merges_path: Path, vocab_path: Path, num_bins: int = 50) -> dict[str, object]:
    merges_raw = json.loads(merges_path.read_text(encoding="utf-8"))
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merges_raw]
    n = len(merges)
    bin_size = max(1, (n + num_bins - 1) // num_bins)

    curve_mean_len = []
    curve_max_len = []
    curve_x = []
    cum_max = 1

    for start in range(0, n, bin_size):
        end = min(n, start + bin_size)
        lengths = [len(a) + len(b) for a, b in merges[start:end]]
        cum_max = max(cum_max, max(lengths))
        curve_x.append((start + end) // 2)
        curve_mean_len.append(round(sum(lengths) / len(lengths), 3))
        curve_max_len.append(cum_max)

    quintiles = []
    q_size = max(1, n // 5)
    cats = ["byte_byte", "ascii_word", "punct_sym", "digit", "non_ascii", "whitespace", "mixed"]
    for qi in range(5):
        s = qi * q_size
        e = n if qi == 4 else (qi + 1) * q_size
        counts = {c: 0 for c in cats}
        for a, b in merges[s:e]:
            counts[_classify_merge(a, b)] += 1
        total_q = max(1, e - s)
        quintiles.append(
            {
                "label": f"{qi * 20}–{(qi + 1) * 20}% (#{s}–#{e - 1})",
                "total": e - s,
                "pct": {c: round(100.0 * counts[c] / total_q, 1) for c in cats},
            }
        )

    first_120 = [
        {
            "rank": i,
            "a": _fmt_bytes(a),
            "b": _fmt_bytes(b),
            "merged": _fmt_bytes(a + b),
            "len": len(a) + len(b),
            "cat": _classify_merge(a, b),
        }
        for i, (a, b) in enumerate(merges[:120])
    ]

    # Sample evenly across 120..n-1
    sampled = []
    if n > 120:
        stride = max(1, (n - 120) // 60)
        for i in range(120, n, stride):
            a, b = merges[i]
            sampled.append(
                {
                    "rank": i,
                    "a": _fmt_bytes(a),
                    "b": _fmt_bytes(b),
                    "merged": _fmt_bytes(a + b),
                    "len": len(a) + len(b),
                    "cat": _classify_merge(a, b),
                }
            )
        # Also include last 15 merges
        for i in range(max(120, n - 15), n):
            a, b = merges[i]
            if not sampled or sampled[-1]["rank"] != i:
                sampled.append(
                    {
                        "rank": i,
                        "a": _fmt_bytes(a),
                        "b": _fmt_bytes(b),
                        "merged": _fmt_bytes(a + b),
                        "len": len(a) + len(b),
                        "cat": _classify_merge(a, b),
                    }
                )

    # Build merge parent lookup to reconstruct binary trees for longest/notable tokens
    parent_map: dict[bytes, tuple[int, bytes, bytes]] = {a + b: (i, a, b) for i, (a, b) in enumerate(merges)}

    def build_tree(tok: bytes) -> dict[str, object]:
        node: dict[str, object] = {"label": _fmt_bytes(tok), "len": len(tok)}
        info = parent_map.get(tok)
        if info is not None:
            rank, left, right = info
            node["rank"] = rank
            node["children"] = [build_tree(left), build_tree(right)]
        return node

    vocab_raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab_map = vocab_raw["vocab"] if "vocab" in vocab_raw else vocab_raw
    vocab = {int(k): bytes.fromhex(v) for k, v in vocab_map.items()}
    top_longest = sorted(vocab.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[:10]
    seen_ids = {tid for tid, _ in top_longest}
    alpha_longest = sorted(
        (
            (tid, tok)
            for tid, tok in vocab.items()
            if tid not in seen_ids
            and len(tok) > 1
            and tok.decode("ascii", errors="ignore").lstrip(" ").isalpha()
            and all(b < 128 for b in tok)
        ),
        key=lambda kv: (len(kv[1]), kv[0]),
        reverse=True,
    )[:6]
    longest_trees = [
        {"id": tid, "len": len(tok), "label": _fmt_bytes(tok), "tree": build_tree(tok)}
        for tid, tok in [*top_longest, *alpha_longest]
    ]

    return {
        "vocab_size": len(vocab),
        "num_merges": n,
        "curve_x": curve_x,
        "curve_mean_len": curve_mean_len,
        "curve_max_len": curve_max_len,
        "quintiles": quintiles,
        "first_120": first_120,
        "milestones": sampled,
        "longest_trees": longest_trees,
    }


def _trace_word(tok: Tokenizer, word_bytes: bytes) -> dict[str, object]:
    pieces = [bytes([b]) for b in word_bytes]
    ranks = tok.merge_ranks
    steps = [{"step": 0, "rank": None, "merged": None, "pieces": [_fmt_bytes(p) for p in pieces]}]
    step_no = 1
    while len(pieces) >= 2:
        best_rank = None
        best_pair = None
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            r = ranks.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_pair = pair
        if best_pair is None:
            break
        merged = best_pair[0] + best_pair[1]
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and pieces[i] == best_pair[0] and pieces[i + 1] == best_pair[1]:
                new_pieces.append(merged)
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1
        pieces = new_pieces
        steps.append(
            {
                "step": step_no,
                "rank": best_rank,
                "merged": f"{_fmt_bytes(best_pair[0])} + {_fmt_bytes(best_pair[1])} → {_fmt_bytes(merged)}",
                "pieces": [_fmt_bytes(p) for p in pieces],
            }
        )
        step_no += 1
    return {"final_count": len(pieces), "steps": steps}


def main() -> None:
    ts_dir = ROOT / "artifacts" / "tinystories_bpe"
    owt_dir = ROOT / "artifacts" / "owt_bpe"
    ts_tok = Tokenizer.from_files(ts_dir / "vocab.json", ts_dir / "merges.json", ["<|endoftext|>"])
    owt_tok = Tokenizer.from_files(owt_dir / "vocab.json", owt_dir / "merges.json", ["<|endoftext|>"])

    ts_data = _analyze_merges(ts_dir / "merges.json", ts_dir / "vocab.json")
    owt_data = _analyze_merges(owt_dir / "merges.json", owt_dir / "vocab.json")

    sample_sentences = [
        {
            "title": "经典对比：长科技词与数字",
            "text": "Today cryptocurrencies and telecommunications reached 2025 milestones!",
        },
        {
            "title": "TinyStories 童话叙事句",
            "text": "Once upon a time, Lily's granddaughter congratulated the clever caterpillars.",
        },
        {
            "title": "网页分隔符、代码与多语言 UTF-8",
            "text": "def f(x_1): return 'hello'\n--------------------------------\n你好🙂",
        },
        {
            "title": "复杂新闻词与标点",
            "text": "The disproportionately large unconstitutional debate surprised environmentalists.",
        },
    ]

    sentence_comparisons = []
    for item in sample_sentences:
        pretokens = _PRETOKEN.findall(item["text"])
        words = []
        for pt in pretokens:
            pt_bytes = pt.encode("utf-8")
            words.append(
                {
                    "pretoken": _fmt_bytes(pt_bytes),
                    "bytes_len": len(pt_bytes),
                    "ts": _trace_word(ts_tok, pt_bytes),
                    "owt": _trace_word(owt_tok, pt_bytes),
                }
            )
        ts_total = sum(w["ts"]["final_count"] for w in words)
        owt_total = sum(w["owt"]["final_count"] for w in words)
        sentence_comparisons.append(
            {
                "title": item["title"],
                "text": item["text"],
                "ts_tokens": ts_total,
                "owt_tokens": owt_total,
                "words": words,
            }
        )

    exp_report_path = ROOT / "artifacts" / "tokenizer_experiments" / "report.json"
    exp_summary = json.loads(exp_report_path.read_text(encoding="utf-8")) if exp_report_path.exists() else {}
    owt_report = json.loads((owt_dir / "report.json").read_text(encoding="utf-8"))

    payload = {
        "tinystories": ts_data,
        "owt": owt_data,
        "sentence_comparisons": sentence_comparisons,
        "owt_report_comparison": owt_report.get("comparison_vs_tinystories", {}),
        "experiments_summary": exp_summary,
    }

    out_path = ROOT / "artifacts" / "tokenizer_experiments" / "viz_data.json"
    if len(sys.argv) > 1:
        out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False)
    out_path.write_text(serialized, encoding="utf-8")
    print(f"Wrote viz data ({out_path.stat().st_size / 1024:.1f} KB) to {out_path}")

    if len(sys.argv) > 2:
        html_path = Path(sys.argv[2])
        html = html_path.read_text(encoding="utf-8")
        placeholder = '"__VIZ_DATA_PLACEHOLDER__"'
        assert placeholder in html, f"Placeholder {placeholder} not found in {html_path}"
        safe_json = serialized.replace("</", "<\\/")
        html_path.write_text(html.replace(placeholder, safe_json, 1), encoding="utf-8")
        print(f"Injected viz data into {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
