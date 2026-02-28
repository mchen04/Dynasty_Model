import torch
import torch.nn as nn


class Stage1Ensembler(nn.Module):
    """
    Blends the discrete cross-sectional predictions from the Quantile Regressor
    with the longitudinal trajectory embeddings from the Time-Series Transformer.

    This ensures we don't just rely on "last year's stats" (Quantile) but also factor in
    whether the player is on a rising breakout trajectory or a declining washout trajectory (Transformer).
    """

    def __init__(self, c_dim=3, t_dim=16, alpha_trainable=True):
        """
        c_dim: number of outputs from cross-sectional model (e.g., Floor, Median, Ceiling)
        t_dim: number of dimensions from transformer sequence embedding
        alpha_trainable: whether to dynamically learn the blend weight between the two models
        """
        super(Stage1Ensembler, self).__init__()

        # We can either use a fixed math blend, or a neural layer to intelligently combine them
        self.combiner = nn.Sequential(
            nn.Linear(c_dim + t_dim, 16),
            nn.ReLU(),
            nn.Linear(16, c_dim),  # Outputs the finalized Floor, Median, Ceiling
        )

        # Alternatively, a simple learned scalar blend: Output = alpha*C + (1-alpha)*T_projected
        self.alpha = nn.Parameter(torch.tensor(0.5)) if alpha_trainable else 0.5
        self.t_projector = nn.Linear(t_dim, c_dim)

    def forward(
        self, cross_sectional_preds, transformer_embeddings, use_neural_blend=False
    ):
        """
        cross_sectional_preds: (Batch, 3) -> [Floor, Median, Ceiling] from Quantile Regressor
        transformer_embeddings: (Batch, 16) -> Trajectory vector from Time-Series Transformer
        """
        if use_neural_blend:
            # Concatenate and pass through MLP
            combined = torch.cat([cross_sectional_preds, transformer_embeddings], dim=1)
            final_predictions = self.combiner(combined)
            return final_predictions
        else:
            # Simple weighted alpha blend
            # First map the 16-D trajectory into the 3-D output space
            t_mapped = self.t_projector(transformer_embeddings)

            # Blend
            final_predictions = (self.alpha * cross_sectional_preds) + (
                (1 - self.alpha) * t_mapped
            )
            return final_predictions


if __name__ == "__main__":
    print("Testing Stage 1 ML Ensembler...")

    batch_size = 4

    # 1. Output from Quantile Regressor (e.g., predicting T+1 Minutes Share)
    # [10th_Floor, 50th_Median, 90th_Ceiling]
    mock_quantile_preds = torch.tensor(
        [
            [0.10, 0.25, 0.35],  # Player A (Role player)
            [0.40, 0.50, 0.55],  # Player B (Starter)
            [0.60, 0.70, 0.75],  # Player C (Star)
            [0.05, 0.15, 0.40],  # Player D (High variance rookie)
        ]
    )

    # 2. Output from Transformer (16-D sequence embedding)
    mock_transformer_embeds = torch.rand(batch_size, 16)

    # Initialize Ensembler
    ensembler = Stage1Ensembler(c_dim=3, t_dim=16, alpha_trainable=True)

    # Blend them!
    final_outputs = ensembler(
        mock_quantile_preds, mock_transformer_embeds, use_neural_blend=True
    )

    print("\nOriginal Quantile Estimations (Cross-Sectional Only):")
    print(mock_quantile_preds)

    print("\nFinal Ensembled Predictions (Blended with Career Trajectory Context):")
    print(final_outputs.detach())
    print(
        "\nEnsemble successful. Trajectory features altered the base quantile expectations."
    )
