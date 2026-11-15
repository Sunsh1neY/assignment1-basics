import heapq
import json
import multiprocessing
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PRETOKEN = re.compile(PAT)
_BYTES = tuple(bytes([i]) for i in range(256))


def _specials(tokens):
    tokens = list(dict.fromkeys(tokens or []))
    if any(not token for token in tokens):
        raise ValueError("Special tokens must be nonempty")
    return sorted(tokens, key=len, reverse=True)


def get_pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    specials = _specials(special_tokens)
    chunks = re.split("|".join(map(re.escape, specials)), text) if specials else [text]
    counts = Counter()
    for chunk in chunks:
        for match in _PRETOKEN.finditer(chunk):
            counts[tuple(_BYTES[b] for b in match.group().encode("utf-8"))] += 1
    return counts


def count_chunk_tokens(args: tuple[str | os.PathLike, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    return get_pretoken_counts(text, special_tokens)


def get_pretoken_counts_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 4,
) -> Counter[tuple[bytes, ...]]:
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    specials = _specials(special_tokens)
    size = os.path.getsize(input_path)
    if num_processes == 1 or not specials:
        return count_chunk_tokens((input_path, 0, size, specials))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, specials[0].encode("utf-8"))
    tasks = [(input_path, start, end, specials) for start, end in zip(boundaries, boundaries[1:])]
    if len(tasks) <= 1:
        return count_chunk_tokens((input_path, 0, size, specials))
    counts = Counter()
    with multiprocessing.Pool(min(num_processes, len(tasks))) as pool:
        for result in pool.imap_unordered(count_chunk_tokens, tasks):
            counts.update(result)
    return counts


class _PairPriority:
    def __init__(self, count, pair):
        self.count = count
        self.pair = pair

    def __lt__(self, other):
        return (self.count, self.pair) > (other.count, other.pair)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    specials = list(dict.fromkeys(special_tokens))
    _specials(specials)
    vocab = dict(enumerate(_BYTES))
    for token in specials:
        if token.encode("utf-8") not in vocab.values():
            vocab[len(vocab)] = token.encode("utf-8")
    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than the initial vocabulary")
    if num_processes is None:
        num_processes = min(4, os.cpu_count() or 1) if os.path.getsize(input_path) >= 8 * 1024**2 else 1
    counts = get_pretoken_counts_parallel(input_path, specials, num_processes)
    words = list(counts)
    frequencies = list(counts.values())
    pair_counts = Counter()
    locations = defaultdict(set)
    for index, word in enumerate(words):
        for pair, count in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += count * frequencies[index]
            locations[pair].add(index)
    heap = [_PairPriority(count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(vocab) < vocab_size and heap:
        entry = heapq.heappop(heap)
        pair = entry.pair
        if entry.count != pair_counts[pair] or entry.count <= 0:
            continue
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged
        changed = set()
        for index in list(locations[pair]):
            word = words[index]
            old_pairs = Counter(zip(word, word[1:]))
            result = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                    result.append(merged)
                    i += 2
                else:
                    result.append(word[i])
                    i += 1
            words[index] = tuple(result)
            new_pairs = Counter(zip(result, result[1:]))
            for old_pair in old_pairs:
                locations[old_pair].discard(index)
            for new_pair in new_pairs:
                locations[new_pair].add(index)
            for affected in old_pairs.keys() | new_pairs.keys():
                pair_counts[affected] += (new_pairs[affected] - old_pairs[affected]) * frequencies[index]
                changed.add(affected)
        for affected in changed:
            if pair_counts[affected] > 0:
                heapq.heappush(heap, _PairPriority(pair_counts[affected], affected))
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.special_tokens = _specials(special_tokens)

        existing_bytes = set(self.vocab.values())
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in existing_bytes:
                self.vocab[max(self.vocab, default=-1) + 1] = st_bytes
                existing_bytes.add(st_bytes)

        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(merges)}
        self.special_tokens_set = set(self.special_tokens)
        self.merges = list(merges)
        self.special_pattern = (
            re.compile("|".join(map(re.escape, self.special_tokens))) if self.special_tokens else None
        )

    def save(self, vocab_filepath, merges_filepath):
        Path(vocab_filepath).write_text(
            json.dumps(
                {"vocab": {str(k): v.hex() for k, v in self.vocab.items()}, "special_tokens": self.special_tokens}
            ),
            encoding="utf-8",
        )
        Path(merges_filepath).write_text(json.dumps([[a.hex(), b.hex()] for a, b in self.merges]), encoding="utf-8")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab_data = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merge_data = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        if "vocab" in vocab_data:
            if special_tokens is None:
                special_tokens = vocab_data["special_tokens"]
            vocab_data = vocab_data["vocab"]
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_data.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in merge_data]
        return cls(vocab, merges, special_tokens)

    def _encode_word(self, token_bytes: bytes) -> list[int]:
        word = [bytes([b]) for b in token_bytes]
        while len(word) >= 2:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if best_pair not in self.merge_ranks:
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        return [self.bytes_to_id[b] for b in word]

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(tok) for tok in sorted_special]
            pattern = f"({'|'.join(escaped)})"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens_set:
                ids.append(self.bytes_to_id[part.encode("utf-8")])
            else:
                for match in re.finditer(PAT, part):
                    token_bytes = match.group().encode("utf-8")
                    ids.extend(self._encode_word(token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            pending += chunk
            spans = []
            start = 0
            if self.special_pattern:
                for match in self.special_pattern.finditer(pending):
                    spans.extend(
                        (start + m.start(), start + m.end()) for m in _PRETOKEN.finditer(pending[start : match.start()])
                    )
                    spans.append(match.span())
                    start = match.end()
            spans.extend((start + m.start(), start + m.end()) for m in _PRETOKEN.finditer(pending[start:]))
            cut = spans[-2][0] if len(spans) >= 2 else 0
            for special in self.special_tokens:
                for length in range(1, min(len(special), len(pending) + 1)):
                    if pending.endswith(special[:length]):
                        prefix_start = len(pending) - length
                        for left, right in spans:
                            if left <= prefix_start < right:
                                cut = min(cut, left)
                                break
            if cut:
                yield from self.encode(pending[:cut])
                pending = pending[cut:]
        yield from self.encode(pending)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")
