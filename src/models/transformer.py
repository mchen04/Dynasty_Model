import math

import torch
import torch.nn as nn

from src.utils.config import load_config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=15):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Stage 1 - Longitudinal Trajectory Processor

    Processes player historical z-scored performance sequences using
    multi-head self-attention. Outputs a 64-dim trajectory embedding
    that feeds into the ensembler.
    """

    def __init__(
        self,
        num_features,
        d_model=None,
        nhead=None,
        num_layers=None,
        dropout=None,
        max_seq_len=None,
        embedding_dim=None,
    ):
        super().__init__()
        config = load_config()
        t_cfg = config["model"]["transformer"]

        self.d_model = d_model or t_cfg["d_model"]
        self.max_seq_len = max_seq_len or t_cfg["max_seq_len"]
        nhead = nhead or t_cfg["nhead"]
        num_layers = num_layers or t_cfg["num_layers"]
        dropout = dropout if dropout is not None else t_cfg["dropout"]
        embedding_dim = embedding_dim or t_cfg["embedding_dim"]

        self.input_projection = nn.Linear(num_features, self.d_model)

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout, max_len=self.max_seq_len
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, embedding_dim),
        )

    def forward(self, src, src_key_padding_mask=None):
        """
        src: (batch_size, seq_len, num_features)
        src_key_padding_mask: (batch_size, seq_len) - True for padded positions
        Returns: (batch_size, embedding_dim) trajectory embedding
        """
        x = self.input_projection(src)  # (batch, seq, d_model)

        # Positional encoding expects (seq, batch, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # back to (batch, seq, d_model)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Extract last valid position using padding mask
        if src_key_padding_mask is not None:
            # ~mask gives valid positions; find the last valid index per batch
            valid_mask = ~src_key_padding_mask  # True = valid
            # Get index of last valid position
            # Use argmax on reversed valid_mask to find last True
            seq_len = valid_mask.shape[1]
            last_valid_idx = seq_len - 1 - valid_mask.flip(dims=[1]).float().argmax(dim=1)
            last_valid_idx = last_valid_idx.long()

            batch_idx = torch.arange(output.size(0), device=output.device)
            final_repr = output[batch_idx, last_valid_idx, :]
        else:
            final_repr = output[:, -1, :]

        trajectory_embedding = self.output_head(final_repr)
        return trajectory_embedding


if __name__ == "__main__":
    print("Testing Time-Series Transformer...")

    batch_size = 2
    seq_len = 5
    num_features = 10

    mock_input = torch.rand(batch_size, seq_len, num_features)

    # Player 1: 5 valid seasons, Player 2: 3 valid (2 padded at start)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [True, True, False, False, False]]
    )

    model = TimeSeriesTransformer(num_features=num_features, max_seq_len=5, embedding_dim=64)

    embeddings = model(mock_input, src_key_padding_mask=padding_mask)

    print(f"Input Shape: {mock_input.shape}")
    print(f"Embedding Shape: {embeddings.shape} (expected: [2, 64])")
    print("Transformer forward pass successful.")
