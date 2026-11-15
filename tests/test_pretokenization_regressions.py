import io

import pytest

from cs336_basics.pretokenization import (
    Tokenizer,
    get_pretoken_counts,
    get_pretoken_counts_parallel,
    train_bpe,
)
from cs336_basics.pretokenization_example import find_chunk_boundaries
from .test_tokenizer import get_tokenizer_from_vocab_merges_path, VOCAB_PATH, MERGES_PATH


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "a<s>xb<s><s> hello",
        "I'll we're abc 123!!",
        "hello   \n\n world\t ",
        "中文🙂hello<s>xbye",
        "   \n\t",
        "<s>x<s>",
    ],
)
def test_stream_all_split_points(text):
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH, ["<s>", "<s>x"])
    expected = tokenizer.encode(text)
    for i in range(len(text) + 1):
        for j in range(i, len(text) + 1):
            assert list(tokenizer.encode_iterable([text[:i], text[i:j], text[j:]])) == expected
    assert list(tokenizer.encode_iterable(iter(text))) == expected


def test_overlapping_training_specials():
    assert get_pretoken_counts("a<s>xb", ["<s>", "<s>x"]) == get_pretoken_counts("a<s>xb", ["<s>x", "<s>"])


def test_serialization(tmp_path):
    vocab = dict(enumerate(bytes([i]) for i in range(256)))
    vocab[256] = b"ab"
    tokenizer = Tokenizer(vocab, [(b"a", b"b")], ["<s>"])
    vp, mp = tmp_path / "vocab.json", tmp_path / "merges.json"
    tokenizer.save(vp, mp)
    loaded = Tokenizer.from_files(vp, mp)
    assert loaded.vocab == tokenizer.vocab
    assert loaded.merges == tokenizer.merges
    assert loaded.encode("ab中<s>") == tokenizer.encode("ab中<s>")
    assert loaded.decode(loaded.encode("ab中<s>")) == "ab中<s>"


def test_parallel_matches_serial(tmp_path):
    path = tmp_path / "corpus.txt"
    text = "中文🙂 ab\r\n<s>" * 2000
    path.write_bytes(text.encode("utf-8"))
    expected = get_pretoken_counts(text, ["<s>"])
    assert get_pretoken_counts_parallel(path, ["<s>"], 2) == expected
    assert train_bpe(path, 280, ["<s>"], num_processes=2) == train_bpe(path, 280, ["<s>"], num_processes=1)


def test_chunk_delimiter_straddles_read_boundary():
    data = b"a" * 9095 + b"<s>" + b"a" * 902
    boundaries = find_chunk_boundaries(io.BytesIO(data), 2, b"<s>")
    assert boundaries == [0, 9095, len(data)]


def test_invalid_vocab_size(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        train_bpe(path, 10, [])
