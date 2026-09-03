"""PyTorch Spatial-Temporal Transformer and TCN architectures for skeletal sequences."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class SpatialTemporalTransformer(nn.Module):
    """Spatial-Temporal Transformer for multi-joint skeleton sequence classification.

    Input: (B, T, 17, 3) or (B, T, 51)
    Output: Logits for num_classes actions (B, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 51,  # 17 keypoints * 3 (x, y, conf)
        num_classes: int = 10,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        max_len: int = 120,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Joint spatial embedding projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, 17, 3) or (B, T, 51).

        Returns:
            Logits of shape (B, num_classes).
        """
        if x.dim() == 4:
            # Flatten (B, T, 17, 3) -> (B, T, 51)
            b, t, k, c = x.shape
            x = x.view(b, t, k * c)

        # (B, T, d_model)
        feat = self.input_projection(x)
        feat = self.pos_encoder(feat)

        # Transformer encoding across temporal frames
        encoded = self.transformer_encoder(feat)  # (B, T, d_model)

        # Global average pooling across time dimension
        pooled = torch.mean(encoded, dim=1)  # (B, d_model)

        # Class logits
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


class ChausalConv1dBlock(nn.Module):
    """Dilated Causal 1D Convolution block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        res = x if self.downsample is None else self.downsample(x)
        out = self.conv(x)
        # Remove future padding to maintain causal property
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return F.relu(out + res)


class TCNSequenceClassifier(nn.Module):
    """Temporal Convolutional Network for lightweight fast skeleton sequence classification."""

    def __init__(
        self,
        input_dim: int = 51,
        num_classes: int = 10,
        num_channels: Optional[list[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 128, 64]

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                ChausalConv1dBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            b, t, k, c = x.shape
            x = x.view(b, t, k * c)

        # Transpose to (B, C, T) for 1D Conv
        x = x.transpose(1, 2)
        feat = self.network(x)  # (B, C_out, T)
        # Global max pooling across time
        pooled = torch.max(feat, dim=2)[0]
        return self.fc(pooled)
