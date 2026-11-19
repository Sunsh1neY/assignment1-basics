from __future__ import annotations

import os
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    # 1. 计算合法起始位置的总数：y 会向右多访问 1 个 token，所以最远起点只能到 len(dataset) - context_length - 1
    num_possible_starts = len(dataset) - context_length
    if num_possible_starts <= 0:
        raise ValueError(
            f"数据集长度 ({len(dataset)}) 必须严格大于 context_length ({context_length}) 才能构造 (x, y) 对"
        )

    # 2. 在 [0, num_possible_starts) 内均匀随机采样 batch_size 个起始下标
    # 用 torch.randint 可以和主程序的 torch.manual_seed 保持一致，方便实验复现
    starts = torch.randint(0, num_possible_starts, size=(batch_size,)).numpy()

    # 3. 广播构造二维索引表，一次性把长度为 context_length + 1 的连续窗口切出来
    # starts[:, None] 形状: (B, 1)，offsets 形状: (S + 1,) -> indices 形状: (B, S + 1)
    offsets = np.arange(context_length + 1)
    indices = starts[:, None] + offsets

    # 4. 只把当前这批极小的切片（约 B × (S+1) 个元素）从 memmap 读入内存并转为 int64
    # 绝不整库 .astype(np.int64)！同时避免 PyTorch torch.from_numpy 不支持 uint16 的问题
    chunks = np.asarray(dataset[indices], dtype=np.int64)

    # 5. 拆出输入序列 x（前 S 个）与预测目标 y（右移 1 位的后 S 个）
    # 切片视图在内存上不连续（跨行步长差 1 个元素），用 ascontiguousarray 保证各自连续且互不干扰
    x_np = np.ascontiguousarray(chunks[:, :-1])
    y_np = np.ascontiguousarray(chunks[:, 1:])

    # 6. 转成 torch.LongTensor 并搬到指定设备（'cpu'、'cuda:0' 或 'mps'；非法设备如 'cuda:99' 会按预期抛错）
    x = torch.from_numpy(x_np).to(device=device)
    y = torch.from_numpy(y_np).to(device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    # 1. 如果传入的是文件系统路径（str 或 Path），先确保父文件夹已创建；
    # 如果传入的是内存二进制流（如 io.BytesIO），跳过建目录操作
    if isinstance(out, (str, os.PathLike)):
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    # 2. 组装完整恢复所需的 3 个状态字段
    checkpoint = {
        "model_state_dict": model.state_dict(),  # 模型所有 Parameter 与 persistent buffer
        "optimizer_state_dict": optimizer.state_dict(),  # AdamW 的 param_groups 与每个参数的 t / m / v
        "iteration": iteration,  # 已完成/当前所在的训练步数
    }

    # 3. 通过 torch.save 写入路径或文件流
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # 1. 自动识别目标模型当前所在的计算设备（例如当初在 GPU 保存的权重，加载回 CPU 继续跑或评估）
    first_param = next(model.parameters(), None)
    map_location = first_param.device if first_param is not None else None

    # 2. 反序列化读取 checkpoint 字典，并自动映射到模型当前设备
    checkpoint = torch.load(src, map_location=map_location)

    # 3. 恢复模型参数状态与优化器动量/步数状态（in-place 覆盖）
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 4. 返回保存时记录的迭代步数 int，便于外层恢复学习率进度与循环计数
    return int(checkpoint["iteration"])

