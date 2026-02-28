import torch
import torch.nn as nn

from src.utils.config import load_config


class Stage1Ensembler(nn.Module):
    """
    Blends tree quantile predictions with transformer trajectory embeddings.

    Takes:
    - Tree predictions: (batch, num_stats, 3) for floor/median/ceiling per stat
    - Transformer trajectory embedding: (batch, embedding_dim)

    Per-stat learned alpha weights let some stats benefit more from trajectory info.
    Monotonicity enforcement guarantees floor <= median <= ceiling.
    """

    def __init__(self, num_stats=None, embedding_dim=None):
        super().__init__()
        config = load_config()

        if num_stats is None:
            num_stats = len(config["model"]["target_stats"])
        if embedding_dim is None:
            embedding_dim = config["model"]["transformer"]["embedding_dim"]

        self.num_stats = num_stats
        self.embedding_dim = embedding_dim

        # Per-stat alpha weights (sigmoid-bounded to [0, 1])
        self.alpha_logits = nn.Parameter(torch.zeros(num_stats))

        # Trajectory-to-adjustment MLP: embedding -> additive corrections
        self.trajectory_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_stats * 3),  # 3 corrections per stat (floor/median/ceiling)
        )

    def forward(self, tree_preds, trajectory_embedding):
        """
        tree_preds: (batch, num_stats, 3) - [floor, median, ceiling] per stat
        trajectory_embedding: (batch, embedding_dim)
        Returns: (batch, num_stats, 3) - ensembled and monotonicity-enforced
        """
        batch_size = tree_preds.size(0)

        # Per-stat blending alphas
        alphas = torch.sigmoid(self.alpha_logits)  # (num_stats,)
        alphas = alphas.unsqueeze(0).unsqueeze(-1)  # (1, num_stats, 1)

        # Trajectory corrections
        corrections = self.trajectory_mlp(trajectory_embedding)  # (batch, num_stats*3)
        corrections = corrections.view(batch_size, self.num_stats, 3)

        # Blend: alpha * tree_preds + (1-alpha) * (tree_preds + corrections)
        # Simplifies to: tree_preds + (1-alpha) * corrections
        blended = tree_preds + (1 - alphas) * corrections

        # Monotonicity enforcement: sort ascending to guarantee floor <= median <= ceiling
        blended_sorted, _ = torch.sort(blended, dim=-1)

        return blended_sorted


if __name__ == "__main__":
    print("Testing Stage 1 ML Ensembler...")

    batch_size = 4
    num_stats = 14
    embedding_dim = 64

    mock_tree_preds = torch.rand(batch_size, num_stats, 3)
    # Sort to simulate proper floor < median < ceiling
    mock_tree_preds, _ = torch.sort(mock_tree_preds, dim=-1)

    mock_embeddings = torch.rand(batch_size, embedding_dim)

    ensembler = Stage1Ensembler(num_stats=num_stats, embedding_dim=embedding_dim)
    output = ensembler(mock_tree_preds, mock_embeddings)

    print(f"Tree Predictions Shape: {mock_tree_preds.shape}")
    print(f"Trajectory Embedding Shape: {mock_embeddings.shape}")
    print(f"Ensembled Output Shape: {output.shape}")

    # Verify monotonicity
    diffs = output[:, :, 1:] - output[:, :, :-1]
    assert (diffs >= 0).all(), "Monotonicity violated!"
    print("Monotonicity enforced: floor <= median <= ceiling for all stats.")
