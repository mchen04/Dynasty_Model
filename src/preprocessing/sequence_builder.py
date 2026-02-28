import pandas as pd
import numpy as np


def build_player_sequences(
    df, player_id_col="PLAYER_ID", season_col="SEASON", max_seq_length=15
):
    """
    Transforms flattened cross-sectional data into padded Sequences [Batch, Seq_Len, Features]
    for injection into the Time-Series Transformer.

    Handles:
    - Variable length careers (padding to `max_seq_length`)
    - Chronological ordering
    - Extracting a feature matrix vs. target matrix
    """
    print(f"Building temporal sequences for {df[player_id_col].nunique()} players...")

    # Sort chronologically
    df = df.sort_values(by=[player_id_col, season_col]).reset_index(drop=True)

    # Define features vs targets
    # For predicting T+1 stats
    exclude_cols = [player_id_col, season_col, "PLAYER_NAME", "TEAM_ID"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    sequences = []
    player_ids = []

    for player, group in df.groupby(player_id_col):
        # Limit to the max sequence length (e.g., last 15 years of their career)
        career_len = len(group)
        features = group[feature_cols].values

        if career_len > max_seq_length:
            features = features[-max_seq_length:]
            career_len = max_seq_length

        # Create a padded zero matrix (Shape: max_seq_length x num_features)
        padded = np.zeros((max_seq_length, len(feature_cols)))
        # Fill the end of the array (pre-padding)
        padded[-career_len:, :] = features

        sequences.append(padded)
        player_ids.append(player)

    # Shape: (Num_Players, Max_Seq_Len, Num_Features)
    sequence_array = np.array(sequences)

    print(f"Generated sequence tensor with shape: {sequence_array.shape}")
    return sequence_array, feature_cols, player_ids


if __name__ == "__main__":
    print("Testing Temporal Sequence Builder...")

    mock_history = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2],
            "PLAYER_NAME": ["LeBron", "LeBron", "LeBron", "Wemby", "Wemby"],
            "SEASON": [2003, 2004, 2005, 2023, 2024],
            "Z_PTS": [1.2, 1.8, 2.5, 1.5, 2.8],
            "Z_REB": [0.5, 0.8, 1.1, 2.0, 2.5],
            "USG_PCT": [25.0, 28.0, 31.0, 22.0, 29.0],
        }
    )

    # Let's say our Transformer accepts up to 5-year sequences
    seq_tensor, feat_cols, player_list = build_player_sequences(
        mock_history, max_seq_length=5
    )

    print(f"Features: {feat_cols}")
    print(f"\nSequence Array shape: {seq_tensor.shape}")
    print("\nLeBron's Padded Sequence (Notice the two rows of 0 padding at the start):")
    print(seq_tensor[0])
