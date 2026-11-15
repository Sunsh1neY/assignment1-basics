from array import array
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
_CHUNK_TARGET_BYTES = 16 * 1024 * 1024
_FINDALL_MAX_LEN = 1_048_576
_PAIR_SHIFT = 32
_PAIR_MASK = (1 << _PAIR_SHIFT) - 1
_ENCODE_CACHE_LIMIT = 131_072


def _specials(tokens):
    tokens = list(dict.fromkeys(tokens or []))
    if any(not token for token in tokens):
        raise ValueError("Special tokens must be nonempty")
    return sorted(tokens, key=len, reverse=True)


def _compile_special_pattern(specials: list[str]) -> re.Pattern | None:
    if not specials:
        return None
    return re.compile("|".join(map(re.escape, specials)))


def _count_chunk_str(chunk: str, counts: Counter[str]) -> None:
    if len(chunk) <= _FINDALL_MAX_LEN:
        counts.update(_PRETOKEN.findall(chunk))
    else:
        for match in _PRETOKEN.finditer(chunk):
            counts[match.group()] += 1


def _accumulate_str_counts(text: str, special_pattern: re.Pattern | None, counts: Counter[str]) -> None:
    if special_pattern is None:
        _count_chunk_str(text, counts)
        return
    pos = 0
    for match in special_pattern.finditer(text):
        if match.start() > pos:
            _count_chunk_str(text[pos : match.start()], counts)
        pos = match.end()
    if pos < len(text):
        _count_chunk_str(text[pos:], counts)


def _str_counts_to_bytes(str_counts: Counter[str]) -> Counter[tuple[bytes, ...]]:
    byte_counts: Counter[tuple[bytes, ...]] = Counter()
    for token_str, freq in str_counts.items():
        key = tuple(_BYTES[b] for b in token_str.encode("utf-8"))
        byte_counts[key] = byte_counts.get(key, 0) + freq
    return byte_counts


def get_pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    specials = _specials(special_tokens)
    str_counts: Counter[str] = Counter()
    _accumulate_str_counts(text, _compile_special_pattern(specials), str_counts)
    return _str_counts_to_bytes(str_counts)


def _count_chunk_spans_str(args: tuple[str | os.PathLike, list[tuple[int, int]], list[str]]) -> Counter[str]:
    input_path, spans, special_tokens = args
    special_pattern = _compile_special_pattern(_specials(special_tokens))
    str_counts: Counter[str] = Counter()
    with open(input_path, "rb") as f:
        for start, end in spans:
            if end <= start:
                continue
            f.seek(start)
            text = f.read(end - start).decode("utf-8")
            _accumulate_str_counts(text, special_pattern, str_counts)
    return str_counts


def count_chunk_tokens(args: tuple[str | os.PathLike, int, int, list[str]]) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = args
    return _str_counts_to_bytes(_count_chunk_spans_str((input_path, [(start, end)], special_tokens)))


def _get_pretoken_str_counts_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 4,
    chunk_size_bytes: int = _CHUNK_TARGET_BYTES,
) -> Counter[str]:
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    specials = _specials(special_tokens)
    size = os.path.getsize(input_path)
    if size == 0:
        return Counter()
    if not specials:
        return _count_chunk_spans_str((input_path, [(0, size)], specials))

    desired_num_chunks = max(num_processes, (size + chunk_size_bytes - 1) // chunk_size_bytes)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, specials[0].encode("utf-8"))
    spans = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]
    if not spans:
        return Counter()
    if num_processes == 1 or len(spans) == 1:
        return _count_chunk_spans_str((input_path, spans, specials))

    workers = min(num_processes, len(spans))
    batch_loads = [(0, i) for i in range(workers)]
    heapq.heapify(batch_loads)
    batches: list[list[tuple[int, int]]] = [[] for _ in range(workers)]
    for span in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
        load, idx = heapq.heappop(batch_loads)
        batches[idx].append(span)
        heapq.heappush(batch_loads, (load + span[1] - span[0], idx))

    tasks = [(input_path, batch, specials) for batch in batches if batch]
    counts: Counter[str] = Counter()
    with multiprocessing.Pool(len(tasks)) as pool:
        for result in pool.imap_unordered(_count_chunk_spans_str, tasks):
            counts.update(result)
            del result
    return counts


def get_pretoken_counts_parallel(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 4,
) -> Counter[tuple[bytes, ...]]:
    return _str_counts_to_bytes(_get_pretoken_str_counts_parallel(input_path, special_tokens, num_processes))


class _PairPriority:
    __slots__ = ("count", "bytes_pair", "pair_key")

    def __init__(self, count: int, bytes_pair: tuple[bytes, bytes], pair_key: int):
        self.count = count
        self.bytes_pair = bytes_pair
        self.pair_key = pair_key

    def __lt__(self, other: "_PairPriority") -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.bytes_pair > other.bytes_pair


def _count_adjacent_pairs(word: tuple[int, ...]) -> dict[int, int]:
    pairs: dict[int, int] = {}
    if len(word) < 2:
        return pairs
    prev = word[0]
    for curr in word[1:]:
        pk = (prev << _PAIR_SHIFT) | curr
        pairs[pk] = pairs.get(pk, 0) + 1
        prev = curr
    return pairs


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
        token_utf8 = token.encode("utf-8")
        if token_utf8 not in vocab.values():
            vocab[len(vocab)] = token_utf8
    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than the initial vocabulary")
    if num_processes is None:
        num_processes = min(8, os.cpu_count() or 1) if os.path.getsize(input_path) >= 8 * 1024**2 else 1

    str_counts = _get_pretoken_str_counts_parallel(input_path, specials, num_processes)
    words: list[tuple[int, ...]] = []
    frequencies: list[int] = []
    for token_str, freq in str_counts.items():
        words.append(tuple(token_str.encode("utf-8")))
        frequencies.append(freq)
    del str_counts

    token_bytes: list[bytes] = list(_BYTES)
    pair_counts: dict[int, int] = {}
    locations: defaultdict[int, array] = defaultdict(lambda: array("I"))
    for index, (word, freq) in enumerate(zip(words, frequencies)):
        if len(word) < 2:
            continue
        word_pairs = _count_adjacent_pairs(word)
        for pk, count in word_pairs.items():
            pair_counts[pk] = pair_counts.get(pk, 0) + count * freq
            locations[pk].append(index)

    heap = [
        _PairPriority(
            count,
            (token_bytes[pk >> _PAIR_SHIFT], token_bytes[pk & _PAIR_MASK]),
            pk,
        )
        for pk, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size and heap:
        entry = heapq.heappop(heap)
        pk = entry.pair_key
        if entry.count != pair_counts.get(pk, 0) or entry.count <= 0:
            continue

        left_id = pk >> _PAIR_SHIFT
        right_id = pk & _PAIR_MASK
        left_bytes = token_bytes[left_id]
        right_bytes = token_bytes[right_id]
        merged_bytes = left_bytes + right_bytes
        merges.append((left_bytes, right_bytes))
        vocab[len(vocab)] = merged_bytes
        new_id = len(token_bytes)
        token_bytes.append(merged_bytes)

        pair_counts.pop(pk, None)
        affected_indices = locations.pop(pk, ())
        changed: set[int] = set()
        for index in affected_indices:
            word = words[index]
            n = len(word)
            first_match = -1
            for i in range(n - 1):
                if word[i] == left_id and word[i + 1] == right_id:
                    first_match = i
                    break
            if first_match == -1:
                continue
            if n == 2:
                words[index] = (new_id,)
                continue
            freq = frequencies[index]
            old_pairs: dict[int, int] = {}
            prev = word[0]
            for curr in word[1:]:
                pair_k = (prev << _PAIR_SHIFT) | curr
                old_pairs[pair_k] = old_pairs.get(pair_k, 0) + 1
                prev = curr

            result = list(word[:first_match])
            result.append(new_id)
            i = first_match + 2
            while i < n:
                if i + 1 < n and word[i] == left_id and word[i + 1] == right_id:
                    result.append(new_id)
                    i += 2
                else:
                    result.append(word[i])
                    i += 1
            new_word = tuple(result)
            words[index] = new_word
            new_pairs: dict[int, int] = {}
            if len(new_word) >= 2:
                prev = new_word[0]
                for curr in new_word[1:]:
                    pair_k = (prev << _PAIR_SHIFT) | curr
                    new_pairs[pair_k] = new_pairs.get(pair_k, 0) + 1
                    prev = curr

            for p, old_c in old_pairs.items():
                new_c = new_pairs.get(p, 0)
                if new_c != old_c and p != pk:
                    updated = pair_counts.get(p, 0) + (new_c - old_c) * freq
                    if updated > 0:
                        pair_counts[p] = updated
                        changed.add(p)
                    else:
                        pair_counts.pop(p, None)
                        locations.pop(p, None)
                        changed.discard(p)

            for p, new_c in new_pairs.items():
                if p not in old_pairs:
                    locations[p].append(index)
                    pair_counts[p] = pair_counts.get(p, 0) + new_c * freq
                    changed.add(p)

        for p in changed:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(
                    heap,
                    _PairPriority(
                        c,
                        (token_bytes[p >> _PAIR_SHIFT], token_bytes[p & _PAIR_MASK]),
                        p,
                    ),
                )

        if len(heap) > max(32_768, 2 * len(pair_counts)):
            heap = [
                _PairPriority(
                    c,
                    (token_bytes[p >> _PAIR_SHIFT], token_bytes[p & _PAIR_MASK]),
                    p,
                )
                for p, c in pair_counts.items()
                if c > 0
            ]
            heapq.heapify(heap)
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
        self.special_pattern = _compile_special_pattern(self.special_tokens)
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}

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
        cached = self._encode_cache.get(token_bytes)
        if cached is not None:
            return list(cached)

        if len(token_bytes) == 1:
            ids = [self.bytes_to_id[token_bytes]]
        else:
            word = [_BYTES[b] for b in token_bytes]
            merge_ranks = self.merge_ranks
            while len(word) >= 2:
                best_rank = None
                best_pair = None
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    rank = merge_ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break

                merged = best_pair[0] + best_pair[1]
                new_word = []
                i = 0
                n = len(word)
                while i < n:
                    if i + 1 < n and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word

            ids = [self.bytes_to_id[b] for b in word]

        if len(self._encode_cache) < _ENCODE_CACHE_LIMIT:
            self._encode_cache[token_bytes] = tuple(ids)
        return ids

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            escaped = [re.escape(tok) for tok in self.special_tokens]
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
                for match in _PRETOKEN.finditer(part):
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
