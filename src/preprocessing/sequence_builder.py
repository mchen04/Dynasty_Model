import numpy as np
import pandas as pd


def build_player_sequences(
    df, player_id_col="PLAYER_ID", season_col="SEASON", max_seq_length=15,
    feature_cols=None,
):
    """
    Transforms flattened cross-sectional data into padded sequences
    [Batch, Seq_Len, Features] for the Time-Series Transformer.

    Args:
        feature_cols: explicit list of columns to use. When provided, only these
            columns appear in the sequence tensor (avoids target-column leakage).

    Returns:
        sequence_array: (Num_Players, Max_Seq_Len, Num_Features) numpy array
        padding_masks: (Num_Players, Max_Seq_Len) boolean array (True = padded/ignore)
        feature_cols: list of feature column names
        player_ids: list of player IDs in order
        player_row_map: dict mapping player_id -> list of original row indices
    """
    print(f"Building temporal sequences for {df[player_id_col].nunique()} players...")

    df = df.sort_values(by=[player_id_col, season_col]).reset_index(drop=True)

    if feature_cols is None:
        exclude_cols = [player_id_col, season_col, "PLAYER_NAME", "TEAM_ID", "Player", "TEAM", "Tm"]
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64, float, int]]

    sequences = []
    masks = []
    player_ids = []
    player_row_map = {}

    for player, group in df.groupby(player_id_col):
        career_len = len(group)
        features = group[feature_cols].values
        row_indices = group.index.tolist()

        if career_len > max_seq_length:
            features = features[-max_seq_length:]
            row_indices = row_indices[-max_seq_length:]
            career_len = max_seq_length

        # Pre-padding: zeros at the start, data at the end
        padded = np.zeros((max_seq_length, len(feature_cols)))
        padded[-career_len:, :] = features

        # Padding mask: True where padded (positions to ignore)
        mask = np.ones(max_seq_length, dtype=bool)
        mask[-career_len:] = False

        sequences.append(padded)
        masks.append(mask)
        player_ids.append(player)
        player_row_map[player] = row_indices

    sequence_array = np.array(sequences)
    padding_masks = np.array(masks)

    print(f"Generated sequence tensor with shape: {sequence_array.shape}")
    return sequence_array, padding_masks, feature_cols, player_ids, player_row_map


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

    seq_tensor, pad_masks, feat_cols, player_list, row_map = build_player_sequences(
        mock_history, max_seq_length=5
    )

    print(f"Features: {feat_cols}")
    print(f"Sequence Array shape: {seq_tensor.shape}")
    print(f"Padding Masks shape: {pad_masks.shape}")
    print("\nLeBron's Padded Sequence:")
    print(seq_tensor[0])
    print("\nLeBron's Padding Mask (True = padded):")
    print(pad_masks[0])
