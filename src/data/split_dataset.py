import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
HELD_OUT_FRACTION = 0.1


def split_rows_by_video(rows: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every language row of a video lands on the same side; test_size is a fraction of videos."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_index, test_index = next(splitter.split(rows, groups=rows["YT ID"]))
    return rows.iloc[train_index].reset_index(drop=True), rows.iloc[test_index].reset_index(drop=True)


def split_dataset2() -> None:
    df = pd.read_json("dataset/p2_dataset.json")
    train_df, held_out_df = split_rows_by_video(df, HELD_OUT_FRACTION)
    val_df, test_df = split_rows_by_video(held_out_df, 0.5)

    train_df.to_json("dataset/p3_dataset_split.json", index=False)
    val_df.to_json("dataset/p3_dataset_validation.json", index=False)
    test_df.to_json("dataset/p3_dataset_test.json", index=False)
