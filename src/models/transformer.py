import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # We scale by something dependent on d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Stage 1 - Longitudinal Trajectory Processor

    Predicts the likelihood of a player breaking out vs washing out
    by interpreting their historical z-scored performance sequences.
    Replaces the LSTM per the 2026 specs for better long-term attention.
    """

    def __init__(
        self,
        num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        max_seq_len=15,
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. Project input features up to Transformer hidden dimension
        self.input_projection = nn.Linear(num_features, d_model)

        # 2. Inject Positional Encoding (Absolute time steps)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # 3. Transformer Encoder Blocks
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,  # PyTorch 1.9+ supports batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # 4. Final Projection Head (Predicts multi-outputs like Mins_Share, TS%, Usage)
        # We'll output a hidden representation that gets ensemble-blended later
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(
                32, 16
            ),  # 16-dim representation vector of the player's vector trajectory
        )

    def forward(self, src, src_key_padding_mask=None):
        """
        src shape: (batch_size, seq_len, num_features)
        src_key_padding_mask shape: (batch_size, seq_len) - True for padded elements
        """
        # Project features
        x = self.input_projection(src)  # (batch, seq, d_model)

        # Transformer expects (seq_len, batch, d_model) if batch_first=False
        # Since batch_first=True, we keep it as (batch, seq, d_model)

        # Apply Positional Encoding (needs to swap back to seq, batch for pos_encoder)
        # Our pos_encoder assumes (seq, batch, features) internally
        x = x.transpose(0, 1)  # -> (seq, batch, dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # -> (batch, seq, dim)

        # Pass through Transformer
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # We care about the final time step's prediction representation (T+1 context)
        # Assuming the sequence is chronologically ordered and left-padded with 0s
        # (Though we can use the padding mask to find the true last element, normally
        # sequences are right-padded or we grab the last element if all are fixed length)
        # For simplicity in this scaffold, let's grab the last step of the sequence.
        final_step_repr = output[:, -1, :]

        # Project to target
        trajectory_embedding = self.output_head(final_step_repr)
        return trajectory_embedding


if __name__ == "__main__":
    print("Testing Time-Series Transformer Scaffold...")

    # Simulate a batch of 2 players, 5 historical seasons, 10 advanced features
    batch_size = 2
    seq_len = 5
    num_features = 10

    # Random tensor simulating the output of `sequence_builder.py`
    mock_input_seq = torch.rand(batch_size, seq_len, num_features)

    # Player 1 has 5 valid seasons, Player 2 only has 3 (so 2 padded zeros)
    # True = Ignore this position (it's padding)
    padding_mask = torch.tensor(
        [[False, False, False, False, False], [True, True, False, False, False]]
    )

    model = TimeSeriesTransformer(num_features=num_features, max_seq_len=5)

    # Forward pass
    trajectory_embeddings = model(mock_input_seq, src_key_padding_mask=padding_mask)

    print(f"Input Shape: {mock_input_seq.shape} -> (Batch, Seq_Len, Features)")
    print(
        f"Trajectory Output Shape: {trajectory_embeddings.shape} -> (Batch, Embedded_Dims)"
    )
    print("Successfully processed sequence through multi-head self-attention.")
