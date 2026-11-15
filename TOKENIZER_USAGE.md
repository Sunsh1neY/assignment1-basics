# BPE 与 Tokenizer

在项目根目录运行相关测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_train_bpe.py tests/test_tokenizer.py tests/test_pretokenization_regressions.py
```

训练与保存：

```python
from cs336_basics.pretokenization import Tokenizer, train_bpe

if __name__ == "__main__":
    vocab, merges = train_bpe("data/corpus.txt", 10000, ["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    tokenizer.save("vocab.json", "merges.json")
    restored = Tokenizer.from_files("vocab.json", "merges.json")
    assert restored.encode("hello") == tokenizer.encode("hello")
```

Windows 多进程训练须在脚本的 `if __name__ == "__main__":` 保护块中启动。
默认对至少 8 MiB 的文件使用最多 4 个进程；可传入 `num_processes=1` 强制串行。
没有特殊 token 时采用串行路径，以免任意字节分块破坏预分词边界。

保存格式是 UTF-8 JSON。词表文件包含 `vocab`（ID 到十六进制字节串）和
`special_tokens`；合并文件按顺序保存两个十六进制字节串组成的列表。
`from_files` 默认恢复保存的特殊 token，也允许显式传入列表覆盖它们。
此格式与测试夹具的 GPT-2 字符映射格式不同。

`encode_iterable` 延迟输出 token，并保留末尾尚不能确定边界的文本。
内存随输入块和未完成预分词的长度增长；极长且没有边界的单个词仍需缓存。
Windows 跳过两项依赖 Linux `resource` 限制的内存测试。
