from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, size: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = float(epsilon)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return value * scale * self.weight


class Mamba2MarketScanner(nn.Module):
    """Pure PyTorch Mamba-2-style selective state-space market scanner."""

    def __init__(self, market_size: int, *, d_ssm: int = 64, head_size: int = 64, state_size: int = 16, dropout: float = 0.5):
        super().__init__()
        if d_ssm % head_size:
            raise ValueError("d_ssm must divide by head_size")
        self.inner = 2 * int(d_ssm)
        self.head_size = int(head_size)
        self.heads = self.inner // self.head_size
        self.state_size = int(state_size)
        self.input_projection = nn.Linear(market_size, d_ssm, bias=False)
        projection_size = 2 * self.inner + 2 * self.heads * self.state_size + self.heads
        self.ssm_projection = nn.Linear(d_ssm, projection_size, bias=True)
        self.depthwise = nn.Conv1d(self.inner, self.inner, kernel_size=4, padding=3, groups=self.inner)
        self.log_a = nn.Parameter(torch.full((self.heads,), -1.0))
        self.skip = nn.Parameter(torch.ones(self.heads))
        self.norm = RMSNorm(self.inner)
        self.output_projection = nn.Linear(self.inner, market_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = market.shape
        compressed = self.input_projection(market)
        projected = self.ssm_projection(compressed)
        x, gate, b, c, delta = torch.split(
            projected,
            [self.inner, self.inner, self.heads * self.state_size, self.heads * self.state_size, self.heads],
            dim=-1,
        )
        x = F.silu(self.depthwise(x.transpose(1, 2))[..., :steps].transpose(1, 2))
        x = x.view(batch, steps, self.heads, self.head_size)
        b = b.view(batch, steps, self.heads, self.state_size)
        c = c.view(batch, steps, self.heads, self.state_size)
        delta = F.softplus(delta)
        state = market.new_zeros(batch, self.heads, self.head_size, self.state_size)
        outputs = []
        a = torch.exp(self.log_a).view(1, self.heads, 1, 1)
        for index in range(steps):
            decay = torch.exp(-a * delta[:, index].view(batch, self.heads, 1, 1))
            injected = delta[:, index].view(batch, self.heads, 1, 1) * x[:, index].unsqueeze(-1) * b[:, index].unsqueeze(-2)
            state = decay * state + injected
            current = (state * c[:, index].unsqueeze(-2)).sum(dim=-1) + self.skip.view(1, self.heads, 1) * x[:, index]
            outputs.append(current.flatten(1))
        scanned = torch.stack(outputs, dim=1)
        return self.output_projection(self.dropout(self.norm(scanned) * F.silu(gate)))


class PaperFactorGate(nn.Module):
    def __init__(self, market_size: int, factor_size: int, beta: float):
        super().__init__()
        self.factor_size = int(factor_size)
        self.beta = float(beta)
        self.projection = nn.Linear(market_size, factor_size, bias=True)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        return self.factor_size * torch.softmax(self.projection(market_features) / self.beta, dim=-1)


class PaperTemporalAttention(nn.Module):
    """TFA uses unscaled attention over the short time axis, as in the paper."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must divide by heads")
        self.heads = int(heads)
        self.head_size = d_model // heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        batch, steps, _ = normalized.shape
        q = self.q(normalized).view(batch, steps, self.heads, self.head_size).transpose(1, 2)
        k = self.k(normalized).view(batch, steps, self.heads, self.head_size).transpose(1, 2)
        v = self.v(normalized).view(batch, steps, self.heads, self.head_size).transpose(1, 2)
        attention = torch.softmax(q @ k.transpose(-1, -2), dim=-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, steps, -1)
        residual = normalized + self.out(attended)
        normalized2 = self.norm2(residual)
        return normalized2 + self.ffn(normalized2)


class PaperCrossStockAttention(nn.Module):
    """CSA applies scaled attention across stocks independently at each time step."""

    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # [stocks, time, model] -> [time, stocks, model]
        by_time = value.transpose(0, 1)
        normalized = self.norm1(by_time)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        residual = normalized + attended
        normalized2 = self.norm2(residual)
        return (normalized2 + self.ffn(normalized2)).transpose(0, 1)


class PaperStockTemporalDistillation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.projection(value)
        query = projected[:, -1:]
        weights = torch.softmax((query * projected).sum(dim=-1), dim=-1).unsqueeze(-1)
        return (weights * value).sum(dim=1)


class PaperAlignedStockMamba(nn.Module):
    """Architecture-aligned StockMamba for a fixed cross-sectional universe."""

    def __init__(self, factor_size: int, market_size: int, *, d_model: int = 256, temporal_heads: int = 4, stock_heads: int = 2, beta: float = 5.0, dropout: float = 0.5):
        super().__init__()
        self.factor_size = int(factor_size)
        self.market_size = int(market_size)
        self.scanner = Mamba2MarketScanner(market_size, d_ssm=64, head_size=64, state_size=16, dropout=dropout)
        self.gate = PaperFactorGate(market_size, factor_size, beta)
        self.factor_projection = nn.Linear(factor_size, d_model)
        self.tfa = PaperTemporalAttention(d_model, temporal_heads, dropout)
        self.csa = PaperCrossStockAttention(d_model, stock_heads, dropout)
        self.std = PaperStockTemporalDistillation(d_model)
        self.decoder = nn.Linear(d_model, 1)

    @staticmethod
    def positional_encoding(length: int, d_model: int, device, dtype) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model))
        encoded = torch.zeros(length, d_model, device=device, dtype=dtype)
        encoded[:, 0::2] = torch.sin(position * scale)
        encoded[:, 1::2] = torch.cos(position * scale)
        return encoded

    def forward(self, stock_factors: torch.Tensor, market_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if stock_factors.ndim != 3 or market_features.ndim != 3:
            raise ValueError("Expected stock [N,T,D] and market [N,T,M]")
        scanned_market = self.scanner(market_features)
        gates = self.gate(scanned_market)
        encoded = self.factor_projection(stock_factors * gates)
        encoded = encoded + self.positional_encoding(encoded.shape[1], encoded.shape[2], encoded.device, encoded.dtype).unsqueeze(0)
        temporal = self.tfa(encoded)
        cross = self.csa(temporal)
        distilled = self.std(cross)
        return {"prediction": self.decoder(distilled).squeeze(-1), "gates": gates}
