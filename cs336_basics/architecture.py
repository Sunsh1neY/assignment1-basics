import math  # 提供平方根 sqrt 计算，用于初始化标准差与注意力缩放因子 1/sqrt(d_k)
import torch  # PyTorch 核心张量与数值计算库
import torch.nn as nn  # 仅使用规范允许的 nn.Module、nn.Parameter 和 nn.ModuleList 基类/容器
from einops import einsum, rearrange  # 自文档化的张量缩并与维度重排工具（作业强烈推荐）

class Linear(nn.Module):  # 继承 nn.Module，实现无偏置（bias-free）线性变换 y = Wx
    def __init__(
        self,
        in_features: int,  # 输入特征维度 d_in
        out_features: int,  # 输出特征维度 d_out
        device: torch.device | None = None,  # 参数存放设备（如 'cpu' 或 'cuda'）
        dtype: torch.dtype | None = None,  # 参数数据类型（如 torch.float32）
    ) -> None:
        super().__init__()  # 必须先调用父类初始化，PyTorch 才能正确追踪子参数
        self.in_features = in_features  # 记录输入维度（便于调试与形状检查）
        self.out_features = out_features  # 记录输出维度
        # 按行主序直接存储 W（形状 d_out x d_in），而不是 W^T，与官方 state_dict 权重形状严格对齐
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # 按 PDF 3.3.1 规定计算标准差 sigma = sqrt(2 / (d_in + d_out))
        std = math.sqrt(2.0 / (in_features + out_features))
        # 使用截断正态分布原地初始化权重，截断区间为 [-3*std, 3*std]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保留输入任意前置 batch/seq 维度 `...`，对最后一维 d_in 与 W^T 做等价缩并得到 `... d_out`
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):  # 继承 nn.Module，实现词表查表映射
    def __init__(
        self,
        num_embeddings: int,  # 词表大小 vocab_size
        embedding_dim: int,  # 词向量维度 d_model
        device: torch.device | None = None,  # 参数设备
        dtype: torch.dtype | None = None,  # 参数数据类型
    ) -> None:
        super().__init__()  # 初始化 nn.Module 基类
        self.num_embeddings = num_embeddings  # 保存词表大小
        self.embedding_dim = embedding_dim  # 保存嵌入维度
        # 创建 (vocab_size, d_model) 嵌入矩阵并注册为可学习 Parameter
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        # 按 PDF 3.3.1 规定初始化：N(0, 1) 截断在 [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 利用张量整数高级索引取出每个 token ID 对应的行向量，输出形状为 (*token_ids.shape, embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):  # 均方根层归一化（LLaMA / Qwen 标准）
    def __init__(
        self,
        d_model: int,  # 隐藏层特征维度
        eps: float = 1e-5,  # 防止除以零或开方不稳定的极小常数
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # 调用父类构造
        self.eps = eps  # 保存平滑项 epsilon
        # PDF 3.3.1 规定 RMSNorm 的可学习缩放增益 g 全初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype  # 记住原始精度（如 bf16/fp16），结尾还原
        x_fp32 = x.to(torch.float32)  # 强制升精度到 float32，防止 x^2 求和在半精度下溢出
        # 对最后一维 d_model 计算平方均值，加上 eps 后取倒数平方根 rsqrt 更快且稳定
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 逐元素除以 RMS 并乘上可学习增益（gain 也转为 float32 保持同精度运算）
        result = x_fp32 * inv_rms * self.weight.to(torch.float32)
        return result.to(in_dtype)  # 降回输入原本的 dtype 后返回


def silu(x: torch.Tensor) -> torch.Tensor:  # SiLU / Swish 激活函数：x * sigmoid(x)
    # 作业明确允许直接调用 torch.sigmoid 保证指数数值稳定性
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):  # 门控线性单元前馈网络 FFN(x) = W2(SiLU(W1 x) * W3 x)
    def __init__(
        self,
        d_model: int,  # 模型主干隐藏维度
        d_ff: int | None = None,  # 内部升维维度；若未指定则自动计算
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # 初始化 nn.Module
        if d_ff is None:  # 若未显式传入 d_ff（如构造完整 LM 时）
            # 取最接近 (8/3)*d_model 且为 64 倍数的值以利用 GPU Tensor Core（如 512->1344, 1600->4288）
            d_ff = max(64, int(round((8 * d_model / 3) / 64)) * 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 门控分支线性变换（参数名 w1 对齐官方权重）
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # 投影回 d_model（参数名 w2）
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # 值分支线性变换（参数名 w3）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(w1(x)) 作为平滑门控信号，与 w3(x) 逐元素相乘后再经 w2 投影回原维度
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):  # 旋转位置编码（不可学习参数，仅预计算三角表）
    def __init__(
        self,
        theta: float,  # RoPE 底数超参 Θ（常见 10000.0）
        d_k: int,  # 单个注意力头的 query / key 向量维度（必须为偶数）
        max_seq_len: int,  # 支持的最大序列长度，用于预生成查找表
        device: torch.device | None = None,
    ) -> None:
        super().__init__()  # 初始化父类
        if d_k % 2 != 0:  # RoPE 每次将相邻两维组成 2D 向量旋转，维度必须是偶数
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")
        # 公式 theta_{i,k} = i / Theta^((2k-2)/d)，指数对应 [0, 2, 4, ..., d_k-2] / d_k
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inv_freq = 1.0 / (theta ** exponents)  # 每个 2D 子空间的旋转角速度 1 / Theta^(2j/d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # 位置下标 0..max_seq_len-1
        angles = einsum(positions, inv_freq, "seq, half -> seq half")  # 外积得到每个位置每个子空间的旋转角 theta_{i,k}
        # 注册为非持久化 buffer（persistent=False）：随模型 .to(device) 移动，但不写入 state_dict()
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 用 token_positions 在第 0 维（seq）查表切片，并转成与 x 相同的 dtype（如 fp16/bf16/fp32）
        cos = self.cos_cached[token_positions].to(x.dtype)
        sin = self.sin_cached[token_positions].to(x.dtype)
        x_even = x[..., 0::2]  # 取出每对向量的偶数维分量 q_{2k-2}（注意是相邻配对，不是前后对半切）
        x_odd = x[..., 1::2]  # 取出每对向量的奇数维分量 q_{2k-1}
        # 应用 2x2 旋转矩阵：cos*x_e - sin*x_o 与 sin*x_e + cos*x_o
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # 在末维交替堆叠两维并展平末两维，恢复成原本 (..., seq_len, d_k) 形状
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:  # 数值稳定的 softmax 实现
    # 沿目标维度提取最大值并保持维度（values 取出数值张量），利用平移不变性避免 exp 溢出 inf
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)  # 最大值位置变为 exp(0) = 1，其余均 <= 1
    return exp_x / exp_x.sum(dim=dim, keepdim=True)  # 沿同一维度求和归一化为概率分布



def scaled_dot_product_attention(
    Q: torch.Tensor,  # 查询张量 (..., queries, d_k)
    K: torch.Tensor,  # 键张量 (..., keys, d_k)
    V: torch.Tensor,  # 值张量 (..., keys, d_v)
    mask: torch.Tensor | None = None,  # 布尔掩码 (..., queries, keys)，True 表示允许注意，False 屏蔽
) -> torch.Tensor:
    d_k = Q.shape[-1]  # 提取键/查询特征维度用于尺度缩放
    # 计算 QK^T / sqrt(d_k)，`...` 自动兼容 3D (batch,q,d) 和 4D (batch,head,q,d)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:  # 若传入因果掩码或自定义掩码
        # 注意规范定义 True=关注、False=屏蔽，所以把 ~mask（False 的位置）填入 -inf
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)  # 在 keys 维度做归一化得到注意力概率权重，-inf 位置严格变为 0
    return einsum(attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")  # 对 V 加权求和


class CausalMultiHeadSelfAttention(nn.Module):  # 因果多头自注意力（可选配 RoPE，单次矩阵乘算出 QKV）
    def __init__(
        self,
        d_model: int,  # 模型维度
        num_heads: int,  # 注意力头数 h
        rope: RotaryPositionalEmbedding | None = None,  # 可选传入的共享 RoPE 实例
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # 调用父类构造
        if d_model % num_heads != 0:  # 确保每个头均匀分得整数维 d_k = d_v = d_model / h
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model  # 保存模型维度
        self.num_heads = num_heads  # 保存头数
        self.d_k = d_model // num_heads  # 计算单头维度
        # 分开注册为 q_proj/k_proj/v_proj/output_proj 以精确对齐参考 state_dict 的命名规则
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope  # 持有共享 RoPE 引用（若为 None 则不施加位置旋转）

    def forward(
        self,
        x: torch.Tensor,  # 输入隐表示 (..., seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # 可选位置索引 (..., seq_len)
    ) -> torch.Tensor:
        seq_len = x.shape[-2]  # 当前序列长度
        # 按 adapters 要求的单次矩阵乘法优化：把 Q/K/V 权重在行维拼接为 (3*d_model, d_model)
        w_qkv = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        # 一次性乘出 [Q, K, V] 拼接张量，形状 (..., seq_len, 3 * d_model)
        qkv = einsum(x, w_qkv, "... seq d_in, three_d d_in -> ... seq three_d")
        # 拆开 three=3 与 num_heads，并把 head 维移到 seq 之前，得到各自分离的 Q/K/V (..., num_heads, seq, d_k)
        q, k, v = rearrange(qkv, "... seq (three h d_k) -> three ... h seq d_k", three=3, h=self.num_heads)

        if self.rope is not None:  # 仅对 Q 和 K 施加 RoPE（绝不对 V 使用 RoPE）
            if token_positions is None:  # 若未传位置索引，默认使用 0..seq_len-1
                pos = torch.arange(seq_len, device=x.device)
            elif 2 <= token_positions.ndim < q.ndim - 1:  # 若位置是 (batch, seq)，需补出 head 广播维 (batch, 1, seq)
                pos = token_positions.unsqueeze(-2)
            else:  # 1D (seq,) 或已对齐多维的情况可直接广播
                pos = token_positions
            q = self.rope(q, pos)  # 每个头独立但共用同一套 RoPE 旋转矩阵
            k = self.rope(k, pos)

        # 生成下三角因果掩码：第 i 个 query 仅能看到 j <= i 的 key（未来位置 j > i 为 False）
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        # 调用自定义缩放点积注意力，输出形状 (..., num_heads, seq_len, d_k)
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        # 把多头结果重新拼回最后一维 (..., seq_len, d_model)
        attn_out = rearrange(attn_out, "... h seq d_v -> ... seq (h d_v)")
        return self.output_proj(attn_out)  # 最后经过输出线性层 W_O 融合多头信息


class TransformerBlock(nn.Module):  # Pre-Norm Transformer 基本块：MHA 子层 + SwiGLU FFN 子层
    def __init__(
        self,
        d_model: int,  # 隐藏特征维度
        num_heads: int,  # 注意力头数
        d_ff: int,  # FFN 中间层维度
        max_seq_len: int | None = None,  # 若外部未共享传入 rope 时，用于内部按需创建 RoPE
        theta: float = 10000.0,  # RoPE 底数
        rope: RotaryPositionalEmbedding | None = None,  # 优先接收外层 TransformerLM 传入的共享 RoPE
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # 初始化父类
        if rope is None and max_seq_len is not None:  # 单独实例化 TransformerBlock 时自行构造 RoPE
            rope = RotaryPositionalEmbedding(theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)  # 注意力子层前的 Pre-Norm（键名 ln1）
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)  # 自注意力子层（键名 attn）
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)  # FFN 子层前的 Pre-Norm（键名 ln2）
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)  # 前馈网络子层（键名 ffn）

    def forward(
        self,
        x: torch.Tensor,  # 块输入 (batch, seq_len, d_model)
        token_positions: torch.Tensor | None = None,  # 位置下标
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)  # 第一残差分支：先归一化再注意力，加回纯净残差流
        x = x + self.ffn(self.ln2(x))  # 第二残差分支：先归一化再前馈，加回纯净残差流
        return x


class TransformerLM(nn.Module):  # 完整 Decoder-only Transformer 语言模型
    def __init__(
        self,
        vocab_size: int,  # 词表大小（决定输入 Embedding 与输出 LM Head 行数）
        context_length: int,  # 模型支持的最大上下文长度（决定 RoPE 预缓存长度）
        d_model: int,  # 模型残差流维度
        num_layers: int,  # 堆叠 TransformerBlock 层数
        num_heads: int,  # 每层注意力头数
        d_ff: int,  # 每层 SwiGLU 中间层维度
        rope_theta: float = 10000.0,  # RoPE Θ 超参
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()  # 初始化父类
        self.context_length = context_length  # 保存最大上下文长度
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)  # 词嵌入层（键名 token_embeddings）
        # 全模型只创建并复用一个共享 RoPE 模块，所有层共用同一份预计算 cos/sin buffer
        rope = RotaryPositionalEmbedding(
            theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device
        )
        # 用 nn.ModuleList 注册 N 层 Block（键名会自动生成 layers.0, layers.1, ...），不能用普通 Python list
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, rope=rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)  # Pre-Norm 架构最后必须再做一次最终归一化（键名 ln_final）
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)  # 线性输出头映射回词表 logits（键名 lm_head）

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        seq_len = in_indices.shape[-1]  # 获取当前输入实际序列长度（支持 <= context_length 的截断输入）
        if seq_len > self.context_length:  # 防御性检查，避免 RoPE buffer 越界
            raise ValueError(f"Sequence length ({seq_len}) exceeds context_length ({self.context_length})")
        token_positions = torch.arange(seq_len, device=in_indices.device)  # 自动生成 0..seq_len-1 位置序列
        x = self.token_embeddings(in_indices)  # (batch, seq_len) -> (batch, seq_len, d_model)
        for layer in self.layers:  # 逐层流经 TransformerBlock
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)  # 最终层归一化，稳定 logits 尺度
        return self.lm_head(x)  # 输出未归一化的 next-token logits，形状 (batch, seq_len, vocab_size)